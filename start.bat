@echo off
setlocal EnableDelayedExpansion
echo Checking configuration...

set "config_file=%~dp0config.txt"
set "toolkit_path="

:: Check for config file and validate path
if exist "%config_file%" (
    for /f "usebackq tokens=*" %%a in ("%config_file%") do set "toolkit_path=%%a"
    if not exist "!toolkit_path!" (
        echo Previous installation path not found: !toolkit_path!
        goto :show_main_menu
    )
    goto :continue_setup
) else (
    goto :show_main_menu
)

:show_main_menu
echo Welcome to AI-Toolkit Setup
echo.
echo Please choose an option:
echo 1. Install AI-Toolkit in UI directory (%~dp0ai-toolkit)
echo 2. Install AI-Toolkit in different location
echo 3. Enter path to existing AI-Toolkit
echo 4. Exit
echo.
set /p "main_choice=Enter your choice (1-4): "

if "!main_choice!"=="1" (
    set "toolkit_path=%~dp0ai-toolkit"
    :: Remove trailing backslash if present
    if "!toolkit_path:~-1!"=="\" set "toolkit_path=!toolkit_path:~0,-1!"
    goto :verify_installation
)
if "!main_choice!"=="2" goto :toolkit_input
if "!main_choice!"=="3" (
    set /p "toolkit_path=Enter path to existing AI-Toolkit: "
    if exist "!toolkit_path!" (
        echo !toolkit_path!> "%config_file%"
        goto :continue_setup
    ) else (
        echo Error: Path not found: !toolkit_path!
        goto :show_main_menu
    )
)
if "!main_choice!"=="4" exit /b 0
goto :show_main_menu

:toolkit_input
set /p "toolkit_path=Enter AI-Toolkit installation path: "
goto :verify_installation

:verify_installation
:: Check if directory exists and is not empty
if exist "!toolkit_path!" (
    dir /a /b "!toolkit_path!" | findstr "^" > nul && (
        echo.
        echo Warning: Directory is not empty: !toolkit_path!
        echo Would you like to:
        echo 1. Choose a different location
        echo 2. Go back to main menu
        set /p "dir_choice=Enter your choice (1-2): "
        
        if "!dir_choice!"=="1" goto :toolkit_input
        if "!dir_choice!"=="2" goto :show_main_menu
        goto :verify_installation
    )
)

:: Check if git is available
git --version >nul 2>&1
if errorlevel 1 (
    echo Git is not installed. Please install Git from https://git-scm.com/downloads
    echo After installing Git, please run this script again.
    pause
    exit /b 1
)

echo This will install AI-Toolkit to: !toolkit_path!
echo The installation process includes:
echo - Cloning the repository
echo - Initializing submodules
echo - Setting up Python virtual environment
echo - Installing PyTorch and other requirements
echo This may take several minutes.
set /p "confirm=Continue? (Y/N): "

if /i not "!confirm!"=="Y" goto :show_main_menu

echo Creating directory...
mkdir "!toolkit_path!" 2>nul

echo Cloning AI-Toolkit repository...
cd "!toolkit_path!" || (
    echo Error: Cannot change to directory: !toolkit_path!
    goto :clone_failed
)

git clone https://github.com/ostris/ai-toolkit.git . || goto :clone_failed
echo Initializing submodules...
git submodule update --init --recursive || goto :submodule_failed

echo Creating virtual environment...
python -m venv venv || goto :venv_failed

echo Activating virtual environment...
call "!toolkit_path!\venv\Scripts\activate.bat"

echo Installing PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || goto :pytorch_failed

echo Installing requirements...
pip install -r requirements.txt || goto :requirements_failed

:: Save the toolkit path to config file
echo !toolkit_path!> "%config_file%"

:: Create config.yaml with the correct paths
echo Creating config.yaml...
(
    echo python_path: "!toolkit_path!\venv\Scripts\python.exe"
    echo toolkit_path: "!toolkit_path!"
    echo script_path: "!toolkit_path!\gui.py"
) > config.yaml
echo Configuration files created.

echo AI-Toolkit installation completed successfully.
cd ..
goto :continue_setup

:clone_failed
echo Failed to clone repository.
goto :install_failed

:submodule_failed
echo Failed to initialize submodules.
goto :install_failed

:venv_failed
echo Failed to create virtual environment.
goto :install_failed

:pytorch_failed
echo Failed to install PyTorch.
goto :install_failed

:requirements_failed
echo Failed to install requirements.
goto :install_failed

:install_failed
echo Installation failed. Please try again or install manually.
pause
exit /b 1

:continue_setup
:: Activate the virtual environment
call "!toolkit_path!\venv\Scripts\activate.bat"

echo Checking GUI requirements...

:: Check for tkinter first (it's part of Python standard library)
python -c "import tkinter" 2>NUL
if errorlevel 1 (
    echo Warning: tkinter not found in Python installation.
    echo Please install Python with tkinter support or use a different Python distribution.
    echo You can download Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Install other requirements from requirements.txt
if exist "requirements.txt" (
    echo Installing GUI dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Warning: Some GUI dependencies failed to install.
        echo The application may not work correctly.
        pause
    ) else (
        echo GUI dependencies installed successfully.
    )
) else (
    echo Warning: GUI requirements.txt not found.
    echo Installing individual dependencies...
    
    python -c "import yaml" 2>NUL
    if errorlevel 1 (
        echo Installing pyyaml...
        pip install pyyaml
    )

    python -c "import chardet" 2>NUL
    if errorlevel 1 (
        echo Installing chardet...
        pip install chardet
    )
)

echo.
echo Setup completed:
echo - AI-Toolkit installed and configured
echo - GUI dependencies installed
echo - Virtual environment activated
echo.
echo Type 'deactivate' to exit the virtual environment.
echo.

:: Launch the GUI
python gui.py

:: Keep the window open
cmd /k