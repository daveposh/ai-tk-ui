@echo off
setlocal EnableDelayedExpansion

:: Clear any pending input at startup (modified approach)
set /p ="Enter" <nul >nul
set /p ="Enter" <nul >nul

echo Checking configuration...

set "config_file=%~dp0config.yaml"
set "toolkit_path="

:: Check for config file and validate path
if exist "%config_file%" (
    for /f "usebackq tokens=1,* delims=:" %%a in ("%config_file%") do (
        if "%%a"=="toolkit_path" set "toolkit_path=%%b"
        set "toolkit_path=!toolkit_path: =!"
    )
    if not exist "!toolkit_path!" (
        echo Previous installation path not found: !toolkit_path!
        :: Clear input before showing menu (modified approach)
        set /p ="Enter" <nul >nul
        goto :show_main_menu
    )
    goto :continue_setup
) else (
    :: Check if ai-toolkit exists in the default location before showing menu
    set "default_toolkit_path=%~dp0ai-toolkit"
    if exist "!default_toolkit_path!\gui.py" (
        echo Found existing AI-Toolkit installation without configuration.
        echo Checking installation status...
        set "toolkit_path=!default_toolkit_path!"
        goto :verify_existing_installation
    )
    :: Clear input before showing menu
    for /f "tokens=1,* delims==" %%a in ('set /p ".="') do rem
    goto :show_main_menu
)

:show_main_menu
:: Clear input buffer more thoroughly (modified approach)
set /p ="Enter" <nul >nul
set "main_choice="

echo Welcome to AI-Toolkit Setup
echo.
echo Please choose an option:
echo 1. Install AI-Toolkit in UI directory (%~dp0ai-toolkit)
echo 2. Install AI-Toolkit in different location
echo 3. Enter path to existing AI-Toolkit
echo 4. Exit
echo.

:: Get user input
set /p "main_choice=Enter your choice (1-4): "

:: Validate input is a number between 1-4
echo.!main_choice!|findstr /r "^[1-4]$" >nul || (
    echo Invalid choice. Please enter a number between 1 and 4.
    timeout /t 2 >nul
    :: Clear input buffer again before returning to menu
    set /p ="Enter" <nul >nul
    goto :show_main_menu
)

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
        if exist "!toolkit_path!\gui.py" (
            echo Creating config.yaml with existing installation...
            > "%config_file%" (
                echo script_path: "%~dp0gui.py"
                echo toolkit_path: "!toolkit_path!"
                echo python_path: "!toolkit_path!\venv\Scripts\python.exe"
                echo train_script_path: "!toolkit_path!\run.py"
            )
            echo Configuration files created.
            goto :continue_setup
        ) else (
            echo Error: Directory does not appear to be a valid AI-Toolkit installation
            goto :show_main_menu
        )
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
        echo Warning: Directory exists: !toolkit_path!
        echo Would you like to:
        echo 1. Update config.yaml to use this directory
        echo 2. Overwrite this directory
        echo 3. Choose a different location
        echo 4. Go back to main menu
        set /p "dir_choice=Enter your choice (1-4): "
        
        if "!dir_choice!"=="1" (
            :: Check if it looks like a valid ai-toolkit installation or has a venv
            if exist "!toolkit_path!\venv\Scripts\activate.bat" (
                echo Found existing virtual environment.
                :: Create config.yaml
                > "%config_file%" (
                    echo script_path: "%~dp0gui.py"
                    echo toolkit_path: "!toolkit_path!"
                    echo python_path: "!toolkit_path!\venv\Scripts\python.exe"
                    echo train_script_path: "!toolkit_path!\run.py"
                )
                echo Configuration files created.
                goto :continue_setup
            ) else if exist "!toolkit_path!\gui.py" (
                echo Found existing AI-Toolkit installation.
                :: Create config.yaml and continue with venv setup
                > "%config_file%" (
                    echo script_path: "%~dp0gui.py"
                    echo toolkit_path: "!toolkit_path!"
                    echo python_path: "!toolkit_path!\venv\Scripts\python.exe"
                    echo train_script_path: "!toolkit_path!\run.py"
                )
                echo Configuration files created.
                
                echo Creating virtual environment...
                cd "!toolkit_path!"
                python -m venv venv || goto :venv_failed
                echo Activating virtual environment...
                call "!toolkit_path!\venv\Scripts\activate.bat"
                goto :continue_setup
            ) else (
                echo Directory exists but is not a complete AI-Toolkit installation.
                echo Will proceed with installation in this directory...
                set "skip_directory_creation=1"
                goto :check_installation_requirements
            )
        )
        if "!dir_choice!"=="2" goto :confirm_overwrite
        if "!dir_choice!"=="3" goto :toolkit_input
        if "!dir_choice!"=="4" goto :show_main_menu
        goto :verify_installation
    )
)

:confirm_overwrite
echo Warning: This will delete all contents in !toolkit_path!
set /p "confirm_delete=Are you sure you want to continue? (Y/N): "
if /i "!confirm_delete!"=="Y" (
    echo Removing existing directory...
    rd /s /q "!toolkit_path!" 2>nul
    goto :check_installation_requirements
)
goto :verify_installation

:check_installation_requirements
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

:perform_installation
if not defined skip_directory_creation (
    echo Creating directory...
    mkdir "!toolkit_path!" 2>nul
)

echo Cloning AI-Toolkit repository...
cd "!toolkit_path!" || (
    echo Error: Cannot change to directory: !toolkit_path!
    goto :clone_failed
)

if defined skip_directory_creation (
    :: Skip git clone and continue with setup
    goto :continue_setup
)

git clone https://github.com/ostris/ai-toolkit.git . || goto :clone_failed
echo Initializing submodules...
git submodule update --init --recursive || goto :submodule_failed

echo Creating virtual environment...
python -m venv venv || goto :venv_failed

echo Activating virtual environment...
call "!toolkit_path!\venv\\Scripts\\activate.bat"

echo Installing PyTorch...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || goto :pytorch_failed

echo Installing requirements...
pip install -r requirements.txt || goto :requirements_failed

:: Create GUI requirements.txt if it doesn't exist
if not exist "%~dp0requirements.txt" (
    echo Creating GUI requirements.txt...
    (
        echo pyyaml
        echo chardet
    ) > "%~dp0requirements.txt"
)

:: Function to replace single backslashes with double backslashes in the path
set "toolkit_path_yaml=!toolkit_path:\=\/!"

:: Create config.yaml with the correct paths
echo Creating config.yaml...
> "%config_file%" (
    echo script_path: "%~dp0gui.py" | powershell -Command "$input -replace '\\', '/'"
    echo toolkit_path: "%~dp0ai-toolkit" | powershell -Command "$input -replace '\\', '/'"
    echo python_path: "%~dp0ai-toolkit/venv/Scripts/python.exe" | powershell -Command "$input -replace '\\', '/'"
    echo train_script_path: "%~dp0ai-toolkit/run.py" | powershell -Command "$input -replace '\\', '/'"
)
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
call "!toolkit_path!\venv\\Scripts\\activate.bat"

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

:: Run the GUI using the config paths
echo Starting GUI...
python -c "import yaml; cfg=yaml.safe_load(open('config.yaml')); import subprocess; subprocess.run([cfg['python_path'], cfg['script_path']])"

:: End the script after GUI closes
echo GUI closed.
exit /b

:: Add new label for verifying and fixing existing installations
:verify_existing_installation
:: Check for required components
set "missing_components="

if not exist "!toolkit_path!\venv" set "missing_components=1"
if not exist "!toolkit_path!\.git" set "missing_components=1"
if not exist "!toolkit_path!\requirements.txt" set "missing_components=1"

if defined missing_components (
    echo Installation appears incomplete. Missing some components.
    echo Would you like to:
    echo 1. Complete the installation
    echo 2. Start fresh installation
    echo 3. Go back to main menu
    set /p "fix_choice=Enter your choice (1-3): "

    if "!fix_choice!"=="1" (
        :: Complete existing installation
        cd "!toolkit_path!"
        
        :: Create venv if missing
        if not exist "venv" (
            echo Creating virtual environment...
            python -m venv venv || goto :venv_failed
        )
        
        :: Activate venv and install requirements
        echo Activating virtual environment...
        call "!toolkit_path!\venv\\Scripts\\activate.bat"
        
        :: Install PyTorch if needed
        python -c "import torch" 2>NUL || (
            echo Installing PyTorch...
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || goto :pytorch_failed
        )
        
        :: Install requirements if needed
        if exist "requirements.txt" (
            echo Installing requirements...
            pip install -r requirements.txt || goto :requirements_failed
        )
        
        :: Initialize git if needed
        if not exist ".git" (
            echo Initializing git repository...
            git init
            git remote add origin https://github.com/ostris/ai-toolkit.git
            git fetch
            git reset --hard origin/main
            git submodule update --init --recursive || goto :submodule_failed
        )
    )
    if "!fix_choice!"=="2" (
        set "skip_directory_creation=1"
        goto :check_installation_requirements
    )
    if "!fix_choice!"=="3" goto :show_main_menu
    goto :verify_existing_installation
)

:: Create config.yaml since everything is verified/fixed
echo Creating config.yaml...
> "%config_file%" (
    echo script_path: "%~dp0gui.py" | powershell -Command "$input -replace '\\', '/'"
    echo toolkit_path: "%~dp0ai-toolkit" | powershell -Command "$input -replace '\\', '/'"
    echo python_path: "%~dp0ai-toolkit/venv/Scripts/python.exe" | powershell -Command "$input -replace '\\', '/'"
    echo train_script_path: "%~dp0ai-toolkit/run.py" | powershell -Command "$input -replace '\\', '/'"
)
echo Configuration files created.

goto :continue_setup

:: Keep the window open (move this to the very end)
cmd /k