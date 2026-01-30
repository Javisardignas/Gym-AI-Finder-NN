@echo off
setlocal enabledelayedexpansion
setlocal enableextensions

REM ============================================================
REM  GYM AI DEPLOYMENT TOOL - Windows Automation Script
REM  Version: 1.0
REM  Description: Automated deployment for Python and Flutter
REM  Supports: Windows 10/11
REM ============================================================

title GYM AI DEPLOYMENT TOOL
color 0C

REM ============================================================
REM GLOBAL VARIABLES
REM ============================================================
set PYTHON_MIN_VERSION=3.8
set SCRIPT_DIR=%~dp0
set LOG_FILE=%SCRIPT_DIR%gym_ai_deploy.log
set VENV_PATH=C:\gym_env
set PYTHON_EXE=%VENV_PATH%\Scripts\python.exe
set FLUTTER_DIR=%SCRIPT_DIR%flutter
set CONFIG_FILE=%SCRIPT_DIR%config.json
set PUBSPEC_FILE=%SCRIPT_DIR%pubspec.yaml

REM Initialize log (skip if error)
echo. > "%LOG_FILE%" 2>nul
if exist "%LOG_FILE%" (
    call :LOG "========== GYM AI DEPLOYMENT STARTED =========="
    call :LOG "Directory: %SCRIPT_DIR%"
    call :LOG "Time: %date% %time%"
)

REM ============================================================
REM MAIN MENU
REM ============================================================
:MENU
cls
color 0C
echo.
echo    =======================================================
echo               SYSTEM STATUS: BULKING IN PROGRESS
echo    =======================================================
echo.
echo                  /\_/\
echo                ( o . o )          "LIGHT WEIGHT,"
echo                 ^>  w  ^<            "BABY!"
echo              __/       \__
echo          /#####\       /#####\
echo         ^| 80 KG ^|=====^| 80 KG ^|
echo          \#####/       \#####/
echo             ^|             ^|
echo.
echo    =======================================================
echo       GYM AI CONTROL PANEL v1.0
echo    =======================================================
echo.
echo     1. Open Terminal
echo     2. Open UI (Flutter)
echo     3. See History
echo     4. Clear History
echo     5. Exit
echo.
echo     ========================================
echo.
set /p choice="  Select option [1-5]: "

if "%choice%"=="1" goto OPEN_TERMINAL
if "%choice%"=="2" goto OPEN_UI
if "%choice%"=="3" goto VIEW_HISTORY
if "%choice%"=="4" goto CLEAR_HISTORY
if "%choice%"=="5" exit /b 0
goto INVALID_CHOICE

:INVALID_CHOICE
cls
color 0C
echo.
echo     ERROR: Invalid option. Please try again.
echo.
timeout /t 2 >nul
goto MENU

REM ============================================================
REM OPTION 1.1: TRAIN NEURAL NETWORK (Backend)
REM ============================================================
:TRAIN_NEURAL
cls
color 0C
echo.
echo     ========================================
echo     ^|  TRAIN NEURAL NETWORK (Option 1.1) ^|
echo     ========================================
echo.

call :LOG "--- Starting Python Setup ---"

REM Check Python installed
call :CHECK_PYTHON
if errorlevel 1 (
    echo.
    echo     [ERROR] Python could not be installed. Aborting...
    call :LOG "ERROR: Python installation failed"
    timeout /t 5 >nul
    goto MENU
)

REM Verify/create virtual environment
if not exist "%VENV_PATH%" (
    echo.
    echo     [*] Creating virtual environment...
    call :LOG "Creating virtual environment at: %VENV_PATH%"
    python -m venv "%VENV_PATH%" --clear
    if errorlevel 1 (
        echo     [ERROR] Could not create virtual environment
        call :LOG "ERROR: Virtual environment creation failed"
        timeout /t 5 >nul
        goto MENU
    )
    echo     [OK] Virtual environment created
    call :LOG "Virtual environment created successfully"
    
    REM Ensure pip is installed
    echo     [*] Verifying pip installation...
    call "%VENV_PATH%\Scripts\activate.bat"
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    echo     [OK] pip installed correctly
) else (
    echo     [OK] Virtual environment already exists
    call :LOG "Virtual environment already exists"
)

REM Activate virtual environment
call "%VENV_PATH%\Scripts\activate.bat"
call :LOG "Virtual environment activated"

REM Verify pip works correctly
echo.
echo     [*] Verifying pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo     [WARNING] pip not working correctly, reinstalling...
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo     [ERROR] Could not repair pip
        echo     [SOLUTION] Delete 'venv' folder and run again
        call :LOG "ERROR: pip corrupted, remove venv and retry"
        pause
        goto MENU
    )
)

REM Install dependencies
echo.
echo     [*] Installing Python dependencies...
call :LOG "Installing Python dependencies"

REM First upgrade pip silently
python -m pip install --upgrade pip --quiet

REM Install packages one by one
echo     [*] Installing torch...
python -m pip install --no-cache-dir torch --quiet
echo     [*] Installing transformers...
python -m pip install --no-cache-dir transformers --quiet
echo     [*] Installing pandas and scikit-learn...
python -m pip install --no-cache-dir pandas scikit-learn numpy --quiet

if errorlevel 1 (
    echo     [ERROR] Could not install dependencies
    call :LOG "ERROR: Dependency installation failed"
    timeout /t 5 >nul
    goto MENU
)
echo     [+] Dependencies installed successfully
call :LOG "Dependencies installed: torch, transformers, pandas, scikit-learn, numpy"

REM Verify Python script exists
if not exist "%SCRIPT_DIR%nngym_v2.py" (
    echo.
    echo     [ERROR] nngym_v2.py not found
    call :LOG "ERROR: nngym_v2.py not found"
    timeout /t 5 >nul
    goto MENU
)

REM Run main script
echo.
echo     [*] Running nngym_v2.py (Training Mode - Training from scratch)...
echo     ========================================
call :LOG "Running training: nngym_v2.py"
echo.

python "%SCRIPT_DIR%nngym_v2.py" 1 train_from_scratch
if errorlevel 1 (
    echo.
    echo     [ERROR] Python script terminated with errors
    call :LOG "ERROR: Python script failed"
) else (
    echo.
    echo     ========================================
    echo     [+] Execution completed
    call :LOG "Script completed successfully"
)
echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM OPTION 2: OPEN UI (Flutter)
REM ============================================================
:OPEN_UI
cls
color 0C
echo.
echo     ========================================
echo     ^|  OPEN UI FLUTTER (Option 2)       ^|
echo     ========================================
echo.

call :LOG "--- Starting Flutter Setup ---"

REM Ensure model selection before launching UI
if not exist "%SCRIPT_DIR%terminal_mode_v2.py" (
    echo     [ERROR] terminal_mode_v2.py not found
    call :LOG "ERROR: terminal_mode_v2.py not found"
    timeout /t 5 >nul
    goto MENU
)

call :CHECK_PYTHON
if errorlevel 1 (
    echo.
    echo     [ERROR] Python could not be installed. Aborting...
    call :LOG "ERROR: Python installation failed"
    timeout /t 5 >nul
    goto MENU
)

set "RUN_PYTHON=python"
if exist "%PYTHON_EXE%" set "RUN_PYTHON=%PYTHON_EXE%"

echo.
cls
color 0C
echo.
echo     ========================================
echo     ^|  SELECT MODEL (Before UI)         ^|
echo     ========================================
echo.
echo     [*] Selecting model before opening UI...
echo     ========================================
call :LOG "Selecting model before Flutter UI"
%RUN_PYTHON% "%SCRIPT_DIR%terminal_mode_v2.py" --load-model
echo.

REM Check Flutter
call :CHECK_FLUTTER
if errorlevel 1 (
    echo.
    echo     [ERROR] Flutter could not be installed. Aborting...
    call :LOG "ERROR: Flutter installation failed"
    timeout /t 5 >nul
    goto MENU
)

REM Create Flutter project structure if needed
if not exist "%SCRIPT_DIR%lib\main.dart" (
    echo.
    echo     [*] Creating Flutter project structure...
    call :LOG "Creating Flutter structure"
    
    if not exist "%SCRIPT_DIR%lib" mkdir "%SCRIPT_DIR%lib"
    if not exist "%SCRIPT_DIR%assets" mkdir "%SCRIPT_DIR%assets"
    
    REM Create pubspec.yaml
    call :CREATE_PUBSPEC
    
    REM Create main.dart
    call :CREATE_MAIN_DART
    
    echo     [+] Flutter structure created
    call :LOG "Flutter structure created successfully"
)

REM Get Flutter dependencies
echo.
echo     [*] Getting Flutter dependencies...
call :LOG "Downloading Flutter dependencies"
cd /d "%SCRIPT_DIR%"
call flutter pub get >nul 2>&1
if errorlevel 1 (
    echo     [WARNING] Some dependency issues encountered
    call :LOG "WARNING: Flutter dependencies had issues"
)

REM Start Python server in background
echo.
echo     [*] Starting Python server in background...
call :LOG "Starting server in background"
start "GYM AI Server" /B "%PYTHON_EXE%" "%SCRIPT_DIR%servidor_simple.py"
timeout /t 3 /nobreak

REM Enable web mode for Chrome
echo.
echo     [*] Setup Flutter for Web (Chrome)...
echo     ========================================
call :LOG "Starting Flutter UI"
echo.

call flutter run -d chrome

echo.
echo     ========================================
echo     [+] Flutter UI completed
call :LOG "Flutter UI closed"
echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM OPTION 1: OPEN TERMINAL (Submenu)
REM ============================================================
:OPEN_TERMINAL
cls
color 0C
echo.
echo                      /\_/\  
echo                     ( o.o ) 
echo                      ^> ^ ^<
echo                     /^|   ^|\
echo                    (_^|   ^|_)
echo.
echo     ========================================
echo     ^|     TERMINAL SUBMENU (Option 1)   ^|
echo     ========================================
echo.
echo     These are the options shown in Terminal Mode v2:
echo.
echo     1. 🔍 Test search (using current model)
echo     2. 🚀 Train new/improved model
echo     3. 📂 Load different model
echo     4. 💾 View training history
echo     5. 📊 View model evolution
echo     6. ❌ Exit
echo.
echo     (Launching terminal_mode_v2.py ...)
echo.
echo     ========================================
echo.

if not exist "%SCRIPT_DIR%terminal_mode_v2.py" (
    echo     [ERROR] terminal_mode_v2.py not found
    call :LOG "ERROR: terminal_mode_v2.py not found"
    timeout /t 5 >nul
    goto MENU
)

call :CHECK_PYTHON
if errorlevel 1 (
    echo.
    echo     [ERROR] Python could not be installed. Aborting...
    call :LOG "ERROR: Python installation failed"
    timeout /t 5 >nul
    goto MENU
)

REM Verify/create virtual environment
if not exist "%VENV_PATH%" (
    echo.
    echo     [*] Creating virtual environment...
    call :LOG "Creating virtual environment at: %VENV_PATH%"
    python -m venv "%VENV_PATH%" --clear
    if errorlevel 1 (
        echo     [ERROR] Could not create virtual environment
        call :LOG "ERROR: Virtual environment creation failed"
        timeout /t 5 >nul
        goto MENU
    )
    echo     [OK] Virtual environment created
    call :LOG "Virtual environment created successfully"
    
    REM Ensure pip is installed
    echo     [*] Verifying pip installation...
    call "%VENV_PATH%\Scripts\activate.bat"
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    echo     [OK] pip installed correctly
) else (
    echo     [OK] Virtual environment already exists
    call :LOG "Virtual environment already exists"
)

REM Activate virtual environment
call "%VENV_PATH%\Scripts\activate.bat"
call :LOG "Virtual environment activated"

REM Verify pip works correctly
echo.
echo     [*] Verifying pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo     [WARNING] pip not working correctly, reinstalling...
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    if errorlevel 1 (
        echo     [ERROR] Could not repair pip
        echo     [SOLUTION] Delete 'venv' folder and run again
        call :LOG "ERROR: pip corrupted, remove venv and retry"
        pause
        goto MENU
    )
)

REM Install dependencies
echo.
echo     [*] Installing Python dependencies...
call :LOG "Installing Python dependencies"

REM First upgrade pip silently
python -m pip install --upgrade pip --quiet

REM Install packages one by one
echo     [*] Installing torch...
python -m pip install --no-cache-dir torch --quiet
echo     [*] Installing transformers...
python -m pip install --no-cache-dir transformers --quiet
echo     [*] Installing pandas and scikit-learn...
python -m pip install --no-cache-dir pandas scikit-learn numpy --quiet

if errorlevel 1 (
    echo     [ERROR] Could not install dependencies
    call :LOG "ERROR: Dependency installation failed"
    timeout /t 5 >nul
    goto MENU
)
echo     [+] Dependencies installed successfully
call :LOG "Dependencies installed: torch, transformers, pandas, scikit-learn, numpy"

set "RUN_PYTHON=python"
if exist "%PYTHON_EXE%" set "RUN_PYTHON=%PYTHON_EXE%"

cls
color 0C
echo.
echo    =======================================================
echo               SYSTEM STATUS: BULKING IN PROGRESS
echo    =======================================================
echo.
%RUN_PYTHON% "%SCRIPT_DIR%terminal_mode_v2.py"
echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM OPTION 1.2: LOAD TRAINED NEURAL NETWORK
REM ============================================================
:LOAD_MODEL
cls
color 0C
echo.
echo     ========================================
echo     ^| LOAD NEURAL NETWORK (Option 1.2)  ^|
echo     ========================================
echo.

call :LOG "--- Loading trained neural network ---"

REM Check if model exists
if not exist "%SCRIPT_DIR%gym_brain_finetuned.pt" (
    echo     [ERROR] Trained model not found
    echo     Please run option 1 - Train Neural Network
    call :LOG "ERROR: Model file not found"
    echo.
    echo     Press any key to continue...
    pause >nul
    goto MENU
)

REM Check Python
call :CHECK_PYTHON
if errorlevel 1 (
    echo.
    echo     [ERROR] Python could not be installed. Aborting...
    call :LOG "ERROR: Python installation failed"
    timeout /t 5 >nul
    goto MENU
)

REM Activate virtual environment
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
    call :LOG "Virtual environment activated"
) else (
    echo     [ERROR] Virtual environment not found
    echo     Run option 1 to setup Python environment
    timeout /t 5 >nul
    goto MENU
)

echo.     [*] Loading neural network...
echo     ========================================
call :LOG "Loading pre-trained neural network"
echo.

python "%SCRIPT_DIR%nngym_v2.py" 4 load_model_only
if errorlevel 1 (
    echo.
    echo     [ERROR] Could not load the model
    call :LOG "ERROR: Model loading failed"
) else (
    echo.
    echo     ========================================
    echo     [+] Model loaded successfully
    call :LOG "Model loaded successfully"
)
echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM OPTION 1.3: LOAD EXAMPLES
REM ============================================================
:LOAD_EXAMPLES
cls
color 0C
echo.
echo     ========================================
echo     ^|     LOAD EXAMPLES (Option 1.3)    ^|
echo     ========================================
echo.

call :LOG "--- Loading example exercises ---"

REM Check if examples file exists
if not exist "%SCRIPT_DIR%ejemplo_validacion.py" (
    echo     [ERROR] Examples file not found
    echo     File required: ejemplo_validacion.py
    call :LOG "ERROR: Examples file not found"
    echo.
    echo     Press any key to continue...
    pause >nul
    goto MENU
)

REM Check Python
call :CHECK_PYTHON
if errorlevel 1 (
    echo.
    echo     [ERROR] Python could not be installed. Aborting...
    call :LOG "ERROR: Python installation failed"
    timeout /t 5 >nul
    goto MENU
)

REM Activate virtual environment
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
    call :LOG "Virtual environment activated"
) else (
    echo     [ERROR] Virtual environment not found
    echo     Run option 1.1 to setup Python environment
    timeout /t 5 >nul
    goto MENU
)

echo.
echo     [*] Loading validation set and examples...
echo     ========================================
call :LOG "Loading validation set with selectable examples"
echo.

python "%SCRIPT_DIR%ejemplo_validacion.py" load_validation_set interactive

echo.
echo     ========================================
echo     [+] Validation examples loaded successfully
call :LOG "Validation examples loaded"
echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM OPTION 1.4: SEARCH EXERCISE
REM ============================================================
:SEARCH_EXERCISE
cls
color 0C
echo.
echo     ========================================
echo     ^|   SEARCH EXERCISE (Option 1.4)   ^|
echo     ========================================
echo.

call :LOG "--- Starting exercise search ---"

REM Check Python
call :CHECK_PYTHON
if errorlevel 1 (
    echo.
    echo     [ERROR] Python could not be installed. Aborting...
    call :LOG "ERROR: Python installation failed"
    timeout /t 5 >nul
    goto MENU
)

REM Activate virtual environment
if exist "%VENV_PATH%\Scripts\activate.bat" (
    call "%VENV_PATH%\Scripts\activate.bat"
    call :LOG "Virtual environment activated"
) else (
    echo     [ERROR] Virtual environment not found
    echo     Run option 1.1 to setup Python environment
    timeout /t 5 >nul
    goto MENU
)

echo.
echo     [*] Starting exercise search (untrained mode)...
echo     ========================================
call :LOG "Running exercise search without model"
echo.

python "%SCRIPT_DIR%nngym_v2.py" 2 search_no_train

echo.
echo     ========================================
echo     [+] Search completed
call :LOG "Exercise search completed"
echo.
echo     Press any key to continue...
pause >nul
goto MENU

echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM OPTION 3: SEE HISTORY
REM ============================================================
:VIEW_HISTORY
cls
color 0C
echo.
echo     ========================================
echo     ^|     TRAINING HISTORY (Option 3)    ^|
echo     ========================================
echo.

call :LOG "--- Viewing training history ---"

if not exist "%SCRIPT_DIR%training_log.json" (
    echo     [!] No training history found
    echo     Please run option 1 - Train Neural Network
    call :LOG "INFO: No training history available"
) else (
    echo     [OK] Training history found
    echo.
    type "%SCRIPT_DIR%training_log.json"
    echo.
)

echo.
echo     ========================================
echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM OPTION 4: CLEAR HISTORY
REM ============================================================
:CLEAR_HISTORY
cls
color 0C
echo.
echo     ========================================
echo     ^|     CLEAR HISTORY (Option 4)      ^|
echo     ========================================
echo.

call :LOG "--- Clearing training history ---"

if not exist "%SCRIPT_DIR%training_log.json" (
    echo     [!] No training history to clear
    call :LOG "INFO: No training history file found"
) else (
    echo     [*] WARNING: This will delete all training history!
    echo.
    set /p confirm="     Are you sure? (yes/no): "
    
    if /i "%confirm%"=="yes" (
        REM Create empty JSON structure
        (
            echo {
            echo   "trainings": []
            echo }
        ) > "%SCRIPT_DIR%training_log.json"
        
        echo.
        echo     [+] Training history cleared successfully!
        call :LOG "Training history cleared by user"
    ) else (
        echo.
        echo     [*] Operation cancelled
        call :LOG "Clear history operation cancelled"
    )
)

echo.
echo     ========================================
echo.
echo     Press any key to continue...
pause >nul
goto MENU

REM ============================================================
REM HELPER FUNCTIONS
REM ============================================================

REM Function: Write to log
:LOG
if exist "%LOG_FILE%" (
    echo [%date% %time%] %~1 >> "%LOG_FILE%" 2>nul
)
goto :eof

REM Function: Check Python
:CHECK_PYTHON
cls
echo.
echo     [*] Checking for Python 3.8+...
call :LOG "Checking Python"

REM Test if Python actually works (not just Microsoft Store alias)
python --version >nul 2>&1
if errorlevel 1 (
    echo     [!] Python not found or not working. Attempting download...
    call :INSTALL_PYTHON
    exit /b !errorlevel!
)

REM Check if it's the Microsoft Store stub
python -c "import sys" >nul 2>&1
if errorlevel 1 (
    echo     [!] Python is Microsoft Store alias, not real Python. Installing...
    call :INSTALL_PYTHON
    exit /b !errorlevel!
)

REM Verify version
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo     [+] Found: %PYTHON_VERSION%
call :LOG "Python found: %PYTHON_VERSION%"

exit /b 0

REM Function: Install Python
:INSTALL_PYTHON
echo     [*] Installing Python from web...
call :LOG "Installing Python automatically"

REM Skip winget/choco and go directly to manual download
REM (These often fail with Microsoft Store stub or outdated versions)
echo     [!] Skipping package managers (often have issues)
echo     [*] Downloading Python installer directly...
call :DOWNLOAD_PYTHON
exit /b !errorlevel!

REM Function: Download Python
:DOWNLOAD_PYTHON
echo     [*] Downloading Python installer...
set PYTHON_URL=https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe
set PYTHON_INSTALLER=%SCRIPT_DIR%python_installer.exe

REM Download with PowerShell
powershell -Command "^
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; ^
  (New-Object Net.WebClient).DownloadFile('%PYTHON_URL%', '%PYTHON_INSTALLER%')
" 2>nul

if exist "%PYTHON_INSTALLER%" (
    echo     [*] Running Python installer...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1
    if errorlevel 1 (
        echo     [ERROR] Python installation failed
        del "%PYTHON_INSTALLER%"
        exit /b 1
    )
    del "%PYTHON_INSTALLER%"
    
    REM Refresh PATH
    call :REFRESH_ENV
    
    echo     [+] Python installed successfully
    echo.
    echo     [!] IMPORTANT: Please close and reopen this terminal
    echo     [!] Then run this script again
    call :LOG "Python installed automatically"
    pause
    exit /b 0
) else (
    echo     [ERROR] Could not download Python
    echo.
    echo     [!] MANUAL INSTALLATION REQUIRED:
    echo     [!] 1. Go to: https://www.python.org/downloads/
    echo     [!] 2. Download Python 3.11 or higher
    echo     [!] 3. Run installer and CHECK "Add Python to PATH"
    echo     [!] 4. Restart this script
    echo.
    call :LOG "ERROR: Python download failed"
    pause
    exit /b 1
)

REM Function: Check Flutter
:CHECK_FLUTTER
cls
echo.
echo     [*] Checking Flutter...
call :LOG "Checking Flutter"

where flutter >nul 2>&1
if errorlevel 1 (
    echo     [!] Flutter not found. Installing...
    call :INSTALL_FLUTTER
    exit /b !errorlevel!
)

REM Verify version
for /f "tokens=*" %%i in ('flutter --version 2^>^&1') do set FLUTTER_VERSION=%%i
echo     [+] Found: %FLUTTER_VERSION%
call :LOG "Flutter found: %FLUTTER_VERSION%"

exit /b 0

REM Function: Install Flutter
:INSTALL_FLUTTER
echo     [*] Installing Flutter...
call :LOG "Installing Flutter"

REM Create directory
if not exist "%FLUTTER_DIR%" mkdir "%FLUTTER_DIR%"

REM Download with PowerShell
set FLUTTER_URL=https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.16.0-stable.zip
set FLUTTER_ZIP=%SCRIPT_DIR%flutter_installer.zip

echo     [*] Downloading Flutter (this may take a few minutes)...
powershell -Command "^
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; ^
  (New-Object Net.WebClient).DownloadFile('%FLUTTER_URL%', '%FLUTTER_ZIP%')
" 2>nul

if exist "%FLUTTER_ZIP%" (
    echo     [*] Extracting Flutter...
    powershell -Command "Expand-Archive -Path '%FLUTTER_ZIP%' -DestinationPath '%FLUTTER_DIR%' -Force"
    
    if errorlevel 1 (
        echo     [ERROR] Flutter extraction failed
        del "%FLUTTER_ZIP%"
        exit /b 1
    )
    
    del "%FLUTTER_ZIP%"
    
    REM Add Flutter to PATH
    set PATH=%FLUTTER_DIR%\flutter\bin;%PATH%
    call :LOG "Flutter installed at: %FLUTTER_DIR%"
    
    echo     [+] Flutter installed successfully
    exit /b 0
) else (
    echo     [ERROR] Could not download Flutter
    call :LOG "ERROR: Flutter download failed"
    exit /b 1
)

REM Function: Create pubspec.yaml
:CREATE_PUBSPEC
setlocal enabledelayedexpansion
(
echo name: gym_ai_app
echo description: GUI interface for intelligent exercise search
echo publish_to: 'none'
echo version: 1.0.0+1
echo environment:
echo   sdk: '^^3.0.0'
echo dependencies:
echo   flutter:
echo     sdk: flutter
echo   cupertino_icons: ^>^=1.0.0
echo   http: ^>^=1.1.0
echo   intl: ^>^=0.19.0
echo dev_dependencies:
echo   flutter_test:
echo     sdk: flutter
echo flutter:
echo   uses-material-design: true
) > "%SCRIPT_DIR%pubspec.yaml"

call :LOG "pubspec.yaml created"
goto :eof

REM Function: Create main.dart
:CREATE_MAIN_DART
if not exist "%SCRIPT_DIR%lib" mkdir "%SCRIPT_DIR%lib"

(
echo import 'package:flutter/material.dart';
echo.
echo void main(^) {
echo   runApp(const GymAIApp(^^);
echo }
echo.
echo class GymAIApp extends StatelessWidget {
echo   const GymAIApp(^^{Key? key^} : super(key: key^;
echo.
echo   @override
echo   Widget build(BuildContext context^) {
echo     return MaterialApp(
echo       title: 'GYM AI',
echo       debugShowCheckedModeBanner: false,
echo       theme: ThemeData(
echo         primaryColor: const Color(0xFFFF0000^,
echo         useMaterial3: true,
echo         colorScheme: ColorScheme.fromSeed(
echo           seedColor: const Color(0xFFFF0000^,
echo         ^),
echo       ^),
echo       home: const GymAIHomePage(^^,
echo     ^;
echo   }
echo }
echo.
echo class GymAIHomePage extends StatefulWidget {
echo   const GymAIHomePage(^^{Key? key^} : super(key: key^;
echo.
echo   @override
echo   State^<GymAIHomePage^> createState(^) =^> _GymAIHomePageState(^^;
echo }
echo.
echo class _GymAIHomePageState extends State^<GymAIHomePage^> {
echo   final TextEditingController _searchController = TextEditingController(^^;
echo   List^<Map^<String, dynamic^>^> _searchResults = [^];
echo   List^<String^> _searchHistory = [^];
echo.
echo   void _performSearch(String query^) {
echo     if (query.isNotEmpty^) {
echo       setState((^) {
echo         if (!_searchHistory.contains(query^)^) {
echo           _searchHistory.insert(0, query^;
echo           if (_searchHistory.length ^> 10^) {
echo             _searchHistory.removeLast(^^;
echo           }
echo         }
echo         _searchResults = [
echo           {
echo             'name': 'Exercise: !query',
echo             'score': 0.92,
echo             'description': 'Description of exercise found'
echo           },
echo         ];
echo       ^^);
echo     }
echo   }
echo.
echo   @override
echo   Widget build(BuildContext context^) {
echo     return Scaffold(
echo       appBar: AppBar(
echo         title: const Text('GYM AI FINDER'^,
echo         backgroundColor: const Color(0xFF8B0000^,
echo         centerTitle: true,
echo         elevation: 10,
echo       ^),
echo       body: Padding(
echo         padding: const EdgeInsets.all(16.0^,
echo         child: SingleChildScrollView(
echo           child: Column(
echo             children: [
echo               const SizedBox(height: 20^,
echo               TextField(
echo                 controller: _searchController,
echo                 decoration: InputDecoration(
echo                   hintText: 'Describe the exercise...',
echo                   border: OutlineInputBorder(
echo                     borderRadius: BorderRadius.circular(10^,
echo                   ^),
echo                   prefixIcon: const Icon(Icons.fitness_center^,
echo                   prefixIconColor: const Color(0xFFB22222^,
echo                 ^),
echo                 onSubmitted: (value^) ^{
echo                   _performSearch(value^;
echo                 ^},
echo               ^),
echo               const SizedBox(height: 20^,
echo               ElevatedButton(
echo                 onPressed: (^) ^{
echo                   _performSearch(_searchController.text^;
echo                 ^},
echo                 style: ElevatedButton.styleFrom(
echo                   backgroundColor: const Color(0xFFFF0000^,
echo                   padding: const EdgeInsets.symmetric(
echo                     horizontal: 50,
echo                     vertical: 20,
echo                   ^),
echo                   shape: RoundedRectangleBorder(
echo                     borderRadius: BorderRadius.circular(15^,
echo                   ^),
echo                 ^),
echo                 child: const Text(
echo                   'SEARCH EXERCISE',
echo                   style: TextStyle(
echo                     color: Colors.white,
echo                     fontSize: 16,
echo                     fontWeight: FontWeight.bold,
echo                   ^),
echo                 ^),
echo               ^),
echo               const SizedBox(height: 30^,
echo               if (_searchResults.isNotEmpty^)
echo                 Column(
echo                   crossAxisAlignment: CrossAxisAlignment.start,
echo                   children: [
echo                     const Text(
echo                       'Results:',
echo                       style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold^,
echo                     ^),
echo                     const SizedBox(height: 15^,
echo                     ..._searchResults.map((result^) =^>
echo                       Card(
echo                         elevation: 5,
echo                         shape: RoundedRectangleBorder(
echo                           borderRadius: BorderRadius.circular(10^,
echo                         ^),
echo                         child: Container(
echo                           decoration: BoxDecoration(
echo                             borderRadius: BorderRadius.circular(10^,
echo                             border: Border.all(
echo                               color: const Color(0xFFB22222^,
echo                               width: 2,
echo                             ^),
echo                           ^),
echo                           padding: const EdgeInsets.all(15^,
echo                           child: Column(
echo                             crossAxisAlignment: CrossAxisAlignment.start,
echo                             children: [
echo                               Text(
echo                                 result['name']!,
echo                                 style: const TextStyle(
echo                                   fontSize: 16,
echo                                   fontWeight: FontWeight.bold,
echo                                 ^),
echo                               ^),
echo                               const SizedBox(height: 8^,
echo                               Text(
echo                                 'Score: ^${result['score']}',
echo                                 style: const TextStyle(
echo                                   color: Color(0xFFFF0000^,
echo                                   fontWeight: FontWeight.bold,
echo                                 ^),
echo                               ^),
echo                               const SizedBox(height: 8^,
echo                               Text(
echo                                 result['description']!,
echo                                 style: const TextStyle(color: Colors.grey^,
echo                               ^),
echo                             ],
echo                           ^),
echo                         ^),
echo                       ^),
echo                     ^).toList(^^),
echo                   ],
echo                 ^),
echo               if (_searchHistory.isNotEmpty^)
echo                 Column(
echo                   crossAxisAlignment: CrossAxisAlignment.start,
echo                   children: [
echo                     const SizedBox(height: 30^,
echo                     const Text(
echo                       'Search History:',
echo                       style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold^,
echo                     ^),
echo                     const SizedBox(height: 10^,
echo                     Wrap(
echo                       spacing: 8,
echo                       children: _searchHistory
echo                           .map((history^) =^>
echo                       Chip(
echo                         label: Text(history^,
echo                         backgroundColor: const Color(0xFFB22222^,
echo                         labelStyle: const TextStyle(color: Colors.white^,
echo                         onDeleted: (^) ^{
echo                           setState((^) =^> _searchHistory.remove(history^^;
echo                         ^},
echo                       ^)
echo                           .toList(^^),
echo                     ^),
echo                   ],
echo                 ^),
echo             ],
echo           ^),
echo         ^),
echo       ^),
echo     ^;
echo   }
echo.
echo   @override
echo   void dispose(^) {
echo     _searchController.dispose(^^;
echo     super.dispose(^^;
echo   }
echo }
) > "%SCRIPT_DIR%lib\main.dart"

call :LOG "main.dart created"
goto :eof

REM Function: Refresh environment variables
:REFRESH_ENV
powershell -Command "^
  [Environment]::SetEnvironmentVariable('PATH', [Environment]::GetEnvironmentVariable('PATH', 'Machine'), 'Process')
" 2>nul
goto :eof

REM ============================================================
REM END OF SCRIPT
REM ============================================================
endlocal
