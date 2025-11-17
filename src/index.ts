import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { ICommandPalette } from '@jupyterlab/apputils';
import { LabIcon } from '@jupyterlab/ui-components';
import { IFileBrowserFactory } from '@jupyterlab/filebrowser';
import { ChatWidget } from './widget';

// Create a custom chat icon
const chatIcon = new LabIcon({
  name: 'deepagents:chat',
  svgstr: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg>'
});

/**
 * The command IDs used by the plugin.
 */
namespace CommandIDs {
  export const openChat = 'deepagents:open-chat';
}

/**
 * Initialization data for the jupyter-deepagents extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'jupyter-deepagents:plugin',
  description: 'A JupyterLab extension for DeepAgents chat interface',
  autoStart: true,
  optional: [ICommandPalette, IFileBrowserFactory],
  activate: (
    app: JupyterFrontEnd,
    palette: ICommandPalette | null,
    browserFactory: IFileBrowserFactory | null
  ) => {
    console.log('JupyterLab extension jupyter-deepagents is activated!');

    let widget: ChatWidget | null = null;

    // Add command to open chat
    app.commands.addCommand(CommandIDs.openChat, {
      label: 'Deep Agents',
      caption: 'Open DeepAgents chat interface',
      icon: chatIcon,
      execute: () => {
        if (!widget || widget.isDisposed) {
          widget = new ChatWidget(app.shell, browserFactory);
          widget.id = 'deepagents-chat';
          widget.title.label = 'Deep Agents';
          widget.title.icon = chatIcon;
          widget.title.closable = true;
        }

        if (!widget.isAttached) {
          app.shell.add(widget, 'right', { rank: 500 });
        }

        app.shell.activateById(widget.id);
      }
    });

    // Add command to command palette
    if (palette) {
      palette.addItem({
        command: CommandIDs.openChat,
        category: 'Deep Agents'
      });
    }

    console.log('DeepAgents chat interface ready');
  }
};

export default plugin;
