"""Tests for MiMo frontend and planner adapters."""

import pytest

from audio_agent.frontend.mimo_frontend import MimoFrontend
from audio_agent.frontend.model_frontend import FrontendInputFormat
from audio_agent.planner.mimo_planner import MimoPlanner
from audio_agent.planner.model_planner import PlannerInputFormat


class MockMimoFrontend(MimoFrontend):
    """Test double that avoids loading the OpenAI client."""

    def initialize_model(self):
        return {"ready": True}


class MockMimoPlanner(MimoPlanner):
    """Test double that avoids loading the OpenAI client."""

    def initialize_model(self):
        return {"ready": True}


class TestMimoFrontend:
    """Unit tests for MiMo frontend initialization and defaults."""

    def test_default_model(self):
        frontend = MockMimoFrontend()
        assert frontend._model == "mimo-v2.5"

    def test_default_base_url(self):
        frontend = MockMimoFrontend()
        assert frontend._base_url == "https://token-plan-cn.xiaomimimo.com/v1"

    def test_default_api_key_env(self):
        frontend = MockMimoFrontend()
        assert frontend._api_key_env == "MIMO_API_KEY"

    def test_name_property(self):
        frontend = MockMimoFrontend()
        assert frontend.name == "mimo_frontend_mimo-v2.5"

    def test_custom_model(self):
        frontend = MockMimoFrontend(model="custom-omni")
        assert frontend._model == "custom-omni"
        assert frontend.name == "mimo_frontend_custom-omni"

    def test_custom_base_url(self):
        frontend = MockMimoFrontend(base_url="https://custom.mimo.com/v1")
        assert frontend._base_url == "https://custom.mimo.com/v1"

    def test_input_format_is_api_model(self):
        frontend = MockMimoFrontend()
        assert frontend.input_format == FrontendInputFormat.API_MODEL

    def test_custom_temperature_and_max_tokens(self):
        frontend = MockMimoFrontend(temperature=0.7, max_tokens=2048)
        assert frontend._temperature == 0.7
        assert frontend._max_tokens == 2048


class TestMimoPlanner:
    """Unit tests for MiMo planner initialization and defaults."""

    def test_default_model(self):
        planner = MockMimoPlanner()
        assert planner._model == "mimo-v2.5-pro"

    def test_default_base_url(self):
        planner = MockMimoPlanner()
        assert planner._base_url == "https://token-plan-cn.xiaomimimo.com/v1"

    def test_default_api_key_env(self):
        planner = MockMimoPlanner()
        assert planner._api_key_env == "MIMO_API_KEY"

    def test_name_property(self):
        planner = MockMimoPlanner()
        assert planner.name == "mimo_planner_mimo-v2.5-pro"

    def test_custom_model(self):
        planner = MockMimoPlanner(model="custom-pro")
        assert planner._model == "custom-pro"
        assert planner.name == "mimo_planner_custom-pro"

    def test_custom_base_url(self):
        planner = MockMimoPlanner(base_url="https://custom.mimo.com/v1")
        assert planner._base_url == "https://custom.mimo.com/v1"

    def test_input_format_is_api_model(self):
        planner = MockMimoPlanner()
        assert planner.input_format == PlannerInputFormat.API_MODEL

    def test_enable_thinking_default(self):
        planner = MockMimoPlanner()
        assert planner._enable_thinking is False

    def test_enable_thinking_true(self):
        planner = MockMimoPlanner(enable_thinking=True)
        assert planner._enable_thinking is True

    def test_default_response_format_is_json_object(self):
        planner = MockMimoPlanner()
        assert planner._response_format == {"type": "json_object"}

    def test_custom_response_format(self):
        planner = MockMimoPlanner(response_format={"type": "json_schema", "schema": {}})
        assert planner._response_format == {"type": "json_schema", "schema": {}}

    def test_response_format_none_for_free_text(self):
        planner = MockMimoPlanner(response_format=None)
        assert planner._response_format is None


class TestMimoFrontendResponseFormat:
    """Tests for MiMo frontend response format option."""

    def test_default_response_format_is_none(self):
        frontend = MockMimoFrontend()
        assert frontend._response_format is None

    def test_custom_response_format(self):
        frontend = MockMimoFrontend(response_format={"type": "json_object"})
        assert frontend._response_format == {"type": "json_object"}


class TestMimoFactoryFunctions:
    """Tests for factory functions in main.py."""

    @pytest.fixture(scope="class", autouse=True)
    def skip_if_langgraph_missing(self):
        try:
            import langgraph  # noqa: F401
        except ImportError:
            pytest.skip("langgraph not installed")

    def test_create_mimo_frontend_defaults(self):
        from audio_agent.main import create_mimo_frontend

        frontend = create_mimo_frontend()
        assert isinstance(frontend, MimoFrontend)
        assert frontend._model == "mimo-v2.5"
        assert frontend._base_url == "https://token-plan-cn.xiaomimimo.com/v1"

    def test_create_mimo_frontend_custom(self):
        from audio_agent.main import create_mimo_frontend

        frontend = create_mimo_frontend(
            model="custom-omni",
            base_url="https://custom.mimo.com/v1",
            temperature=0.7,
            max_tokens=2048,
        )
        assert frontend._model == "custom-omni"
        assert frontend._base_url == "https://custom.mimo.com/v1"
        assert frontend._temperature == 0.7
        assert frontend._max_tokens == 2048

    def test_create_mimo_planner_defaults(self):
        from audio_agent.main import create_mimo_planner

        planner = create_mimo_planner()
        assert isinstance(planner, MimoPlanner)
        assert planner._model == "mimo-v2.5-pro"
        assert planner._base_url == "https://token-plan-cn.xiaomimimo.com/v1"

    def test_create_mimo_planner_custom(self):
        from audio_agent.main import create_mimo_planner

        planner = create_mimo_planner(
            model="custom-pro",
            base_url="https://custom.mimo.com/v1",
            enable_thinking=True,
            temperature=0.7,
        )
        assert planner._model == "custom-pro"
        assert planner._base_url == "https://custom.mimo.com/v1"
        assert planner._enable_thinking is True
        assert planner._temperature == 0.7
