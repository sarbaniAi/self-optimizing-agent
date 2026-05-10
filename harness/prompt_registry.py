"""
Prompt Registry Manager
=======================
Manage agent prompts via MLflow Prompt Registry with version control
and alias promotion.

Uses MLflow 3.1+ GenAI prompt registry APIs.

Workspace : https://adb-7405619910560146.6.azuredatabricks.net
Catalog   : classic_stable_4a2ohn_azure
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptRegistryManager:
    """
    Manage agent prompts via MLflow Prompt Registry with version control
    and alias promotion.

    Usage
    -----
    >>> mgr = PromptRegistryManager("self_opt_agent_system_prompt")
    >>> version = mgr.register_prompt("You are a helpful assistant.", commit_message="initial")
    >>> mgr.promote_to_production(version, eval_score=0.92, baseline_score=0.88)
    """

    def __init__(self, prompt_name: str, production_alias: str = "production"):
        """
        Parameters
        ----------
        prompt_name      : Logical name in the MLflow Prompt Registry.
        production_alias : Alias that points to the live production version.
        """
        self.prompt_name = prompt_name
        self.production_alias = production_alias

    # -- Load -----------------------------------------------------------------

    def load_prompt(self, version: Optional[str] = None) -> str:
        """
        Load a prompt from the registry.

        Parameters
        ----------
        version : Specific version number (as string) or ``None`` to load
                  the current production alias.

        Returns
        -------
        The prompt template string.
        """
        try:
            import mlflow.genai
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai is required. Install mlflow>=3.1 with GenAI extras."
            ) from exc

        if version is not None:
            uri = f"prompts:/{self.prompt_name}/{version}"
        else:
            uri = f"prompts:/{self.prompt_name}@{self.production_alias}"

        logger.info("Loading prompt from %s", uri)

        try:
            prompt = mlflow.genai.load_prompt(uri)
        except Exception as exc:
            logger.error("Failed to load prompt '%s': %s", uri, exc)
            raise

        # MLflow may return a Prompt object; extract the template string.
        if hasattr(prompt, "template"):
            return prompt.template
        return str(prompt)

    # -- Register -------------------------------------------------------------

    def register_prompt(
        self,
        template: str,
        commit_message: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Register a new prompt version in the registry.

        Parameters
        ----------
        template       : The prompt template string (may contain ``{{variable}}`` placeholders).
        commit_message : Human-readable description of the change.
        tags           : Optional key-value tags to attach to this version.

        Returns
        -------
        The new version number (int).
        """
        try:
            import mlflow.genai
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai is required. Install mlflow>=3.1 with GenAI extras."
            ) from exc

        logger.info(
            "Registering new version of prompt '%s' (message: %s)",
            self.prompt_name,
            commit_message or "<none>",
        )

        try:
            result = mlflow.genai.register_prompt(
                name=self.prompt_name,
                template=template,
                commit_message=commit_message,
            )
        except Exception as exc:
            logger.error("Failed to register prompt: %s", exc)
            raise

        # Extract version number from the result.
        version: int
        if hasattr(result, "version"):
            version = int(result.version)
        elif isinstance(result, dict) and "version" in result:
            version = int(result["version"])
        else:
            logger.warning(
                "Could not extract version number from result (%s). Defaulting to 1.",
                type(result).__name__,
            )
            version = 1

        # Apply optional tags.
        if tags:
            try:
                from mlflow import MlflowClient

                client = MlflowClient()
                for key, value in tags.items():
                    client.set_registered_prompt_tag(
                        name=self.prompt_name, key=key, value=value
                    )
            except Exception as tag_exc:
                logger.warning("Could not set prompt tags: %s", tag_exc)

        logger.info(
            "Prompt '%s' registered as version %d.", self.prompt_name, version
        )
        return version

    # -- Promote --------------------------------------------------------------

    def promote_to_production(
        self,
        version: int,
        eval_score: float,
        baseline_score: float,
    ) -> bool:
        """
        Promote a prompt version to production ONLY if it improves over the baseline.

        Parameters
        ----------
        version        : Version number to promote.
        eval_score     : Evaluation score of the candidate version.
        baseline_score : Score of the current production version.

        Returns
        -------
        True if promotion succeeded, False if gated.
        """
        try:
            import mlflow.genai
        except ImportError as exc:
            raise ImportError(
                "mlflow.genai is required. Install mlflow>=3.1 with GenAI extras."
            ) from exc

        if eval_score <= baseline_score:
            logger.warning(
                "Promotion BLOCKED: candidate score (%.4f) does not exceed "
                "baseline (%.4f). Version %d will NOT be promoted.",
                eval_score,
                baseline_score,
                version,
            )
            return False

        logger.info(
            "Promoting prompt '%s' version %d to alias '%s' "
            "(eval=%.4f > baseline=%.4f).",
            self.prompt_name,
            version,
            self.production_alias,
            eval_score,
            baseline_score,
        )

        try:
            mlflow.genai.set_prompt_alias(
                self.prompt_name, self.production_alias, version
            )
        except Exception as exc:
            logger.error("Failed to set prompt alias: %s", exc)
            raise

        logger.info(
            "Prompt '%s' version %d is now '%s'.",
            self.prompt_name,
            version,
            self.production_alias,
        )
        return True

    # -- History --------------------------------------------------------------

    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        List all prompt versions with metadata.

        Returns
        -------
        List of dicts, each containing version, template (truncated),
        commit_message, tags, and creation timestamp.
        """
        try:
            from mlflow import MlflowClient

            client = MlflowClient()
        except ImportError as exc:
            raise ImportError(
                "MlflowClient is required. Install mlflow>=3.1."
            ) from exc

        logger.info("Fetching version history for prompt '%s'.", self.prompt_name)

        try:
            # MLflow 3.1+ exposes search_registered_prompts / get_registered_prompt
            prompt_info = client.get_registered_prompt(name=self.prompt_name)
        except AttributeError:
            logger.warning(
                "MlflowClient.get_registered_prompt not available. "
                "Returning empty history."
            )
            return []
        except Exception as exc:
            logger.error("Failed to get prompt info: %s", exc)
            raise

        versions: List[Dict[str, Any]] = []
        try:
            all_versions = prompt_info.versions if hasattr(prompt_info, "versions") else []
            for v in all_versions:
                versions.append(
                    {
                        "version": getattr(v, "version", None),
                        "template_preview": (
                            getattr(v, "template", "")[:200] + "..."
                            if len(getattr(v, "template", "")) > 200
                            else getattr(v, "template", "")
                        ),
                        "commit_message": getattr(v, "commit_message", ""),
                        "creation_timestamp": getattr(v, "creation_timestamp", None),
                        "tags": getattr(v, "tags", {}),
                    }
                )
        except Exception as exc:
            logger.warning("Could not iterate prompt versions: %s", exc)

        logger.info("Found %d versions.", len(versions))
        return versions
