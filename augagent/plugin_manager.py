import os
import importlib.util
import logging
from typing import List
from augagent.tools import AugTool

class PluginManager:
    def __init__(self):
        self.plugins = {}
        self.tools = []

    def discover_plugins(self, plugins_dir: str | None = None):
        """Discover and load all plugins from the given directory."""
        if plugins_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plugins_dir = os.path.join(base_dir, "plugins")
            
        if not os.path.exists(plugins_dir):
            return

        for entry in os.listdir(plugins_dir):
            plugin_path = os.path.join(plugins_dir, entry)
            if os.path.isdir(plugin_path) and entry.startswith('augagent-plugin-'):
                plugin_file = os.path.join(plugin_path, "plugin.py")
                if os.path.exists(plugin_file):
                    self.load_plugin(entry, plugin_file)
                
    def load_plugin(self, plugin_name: str, file_path: str):
        try:
            # Create a valid module name from the plugin directory name
            module_name = plugin_name.replace("-", "_")
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec:
                module = importlib.util.module_from_spec(spec)
                if hasattr(spec, "loader") and spec.loader:
                    spec.loader.exec_module(module)
            
            if hasattr(module, 'setup'):
                module.setup(self)
                self.plugins[plugin_name] = module
                logging.info(f"Successfully loaded plugin: {plugin_name}")
            else:
                logging.warning(f"Plugin {plugin_name} is missing a setup(plugin_manager) function.")
        except Exception as e:
            logging.error(f"Failed to load plugin {plugin_name}: {e}")

    def register_tool(self, tool: AugTool):
        """Register a tool provided by a plugin."""
        self.tools.append(tool)

    def get_all_tools(self) -> List[AugTool]:
        """Return all tools registered by plugins."""
        return self.tools

plugin_manager = PluginManager()
