# AI-TK-UI

A graphical user interface for the AI-Toolkit, providing an easy-to-use interface for AI model training and management.

## Prerequisites

- Python 3.10 or higher
- Git
- CUDA compatible GPU (for AI-Toolkit)

## Installation

1. Clone this repository:

2. Run the setup script:
   - Windows: Double-click `start.bat` or run in command prompt:

The setup script will:
- Install or locate AI-Toolkit
- Set up Python virtual environment
- Install required dependencies
- Configure paths and settings

## Features

- Graphical interface for AI-Toolkit operations
- Easy model management and training
- Configuration saving and loading
- Integrated virtual environment management

## Configuration

The application uses several configuration files:
- `config.yaml`: Main configuration file for paths and settings
- `last_settings.json`: Saves your last used settings

## Directory Structure

README.md
ai-tk-ui/
├── gui.py # Main GUI application
├── start.bat # Setup and launch script
├── requirements.txt # Python dependencies
└── ai-toolkit/ # AI-Toolkit installation (managed by start.bat)

## Usage

1. Launch the application using `start.bat`
2. Configure your AI-Toolkit settings
3. Use the GUI to manage and train models

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [AI-Toolkit](https://github.com/ostris/ai-toolkit) - The core toolkit this UI is built for

### Project Structure

- `gui.py`: Main application entry point
- `start.bat`: Environment setup and launcher
- `requirements.txt`: Project dependencies

### Coding Standards

- Follow PEP 8 guidelines
- Include docstrings for functions and classes
- Add comments for complex logic
- Update documentation for new features

## Support

For support, please:
1. Check existing issues on GitHub
2. Create a new issue with:
   - Detailed description of the problem
   - Steps to reproduce
   - System information
   - Error messages

## Roadmap

- [ ] Additional model management features
- [ ] Enhanced training visualization
- [ ] Multi-language support
- [ ] Dark/Light theme support
