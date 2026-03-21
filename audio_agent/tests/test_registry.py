"""Tests for the tool registry."""

import pytest

from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.base import BaseTool
from audio_agent.tools.dummy_tools import DummyASRTool, DummyAudioEventDetectorTool
from audio_agent.core.schemas import ToolSpec, ToolCallRequest, ToolResult
from audio_agent.core.errors import ToolRegistryError


class TestToolRegistry:
    """Tests for ToolRegistry class."""
    
    def test_register_and_get(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()
        tool = DummyASRTool()
        
        registry.register(tool)
        
        retrieved = registry.get("dummy_asr")
        assert retrieved is tool
    
    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        tool1 = DummyASRTool()
        tool2 = DummyAudioEventDetectorTool()
        
        registry.register(tool1)
        registry.register(tool2)
        
        assert len(registry) == 2
        assert "dummy_asr" in registry
        assert "dummy_audio_event_detector" in registry
    
    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises error."""
        registry = ToolRegistry()
        tool = DummyASRTool()
        
        registry.register(tool)
        
        with pytest.raises(ToolRegistryError, match="already registered"):
            registry.register(tool)
    
    def test_unknown_tool_raises(self):
        """Test that looking up unknown tool raises error."""
        registry = ToolRegistry()
        
        with pytest.raises(ToolRegistryError, match="Unknown tool"):
            registry.get("nonexistent")
    
    def test_empty_name_lookup_raises(self):
        """Test that looking up empty name raises error."""
        registry = ToolRegistry()
        
        with pytest.raises(ToolRegistryError, match="empty name"):
            registry.get("")
    
    def test_list_specs(self):
        """Test listing tool specifications."""
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        registry.register(DummyAudioEventDetectorTool())
        
        specs = registry.list_specs()
        
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert names == {"dummy_asr", "dummy_audio_event_detector"}
    
    def test_list_names(self):
        """Test listing tool names."""
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        registry.register(DummyAudioEventDetectorTool())
        
        names = registry.list_names()
        
        assert set(names) == {"dummy_asr", "dummy_audio_event_detector"}
    
    def test_contains(self):
        """Test __contains__ method."""
        registry = ToolRegistry()
        registry.register(DummyASRTool())
        
        assert "dummy_asr" in registry
        assert "nonexistent" not in registry
    
    def test_len(self):
        """Test __len__ method."""
        registry = ToolRegistry()
        assert len(registry) == 0
        
        registry.register(DummyASRTool())
        assert len(registry) == 1
        
        registry.register(DummyAudioEventDetectorTool())
        assert len(registry) == 2
    
    def test_register_non_tool_raises(self):
        """Test that registering non-tool object raises error."""
        registry = ToolRegistry()
        
        with pytest.raises(ToolRegistryError, match="non-tool object"):
            registry.register("not a tool")


class MockToolWithEmptyName(BaseTool):
    """Mock tool with empty name for testing."""
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="",  # This will fail validation
            description="Test",
        )
    
    def invoke(self, request: ToolCallRequest) -> ToolResult:
        return ToolResult(tool_name="", success=True)


class TestToolRegistryEdgeCases:
    """Edge case tests for ToolRegistry."""
    
    def test_empty_tool_name_raises_via_pydantic(self):
        """Test that ToolSpec validation catches empty name."""
        # ToolSpec itself should reject empty names via min_length=1
        with pytest.raises(Exception):  # Pydantic ValidationError
            ToolSpec(name="", description="Bad tool")
    
    def test_whitespace_only_tool_name_raises(self):
        """Test that tool with whitespace-only name cannot be registered."""
        registry = ToolRegistry()
        
        # Create a tool with whitespace name (which passes ToolSpec min_length)
        class WhitespaceTool(BaseTool):
            @property
            def spec(self) -> ToolSpec:
                return ToolSpec(name="   ", description="Whitespace tool")
            
            def invoke(self, request: ToolCallRequest) -> ToolResult:
                pass
        
        # The registry should catch whitespace-only names
        with pytest.raises(ToolRegistryError, match="empty name"):
            registry.register(WhitespaceTool())
