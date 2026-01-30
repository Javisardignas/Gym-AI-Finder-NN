@echo off
title GYM AI - Interfaz Web
color 0A

echo.
echo ================================================
echo          GYM AI - INTERFAZ WEB
echo ================================================
echo.

REM Ruta del script
set SCRIPT_DIR=%~dp0

REM Verificar si existe Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [X] Python no encontrado
    echo     Por favor instala Python 3.8 o superior
    echo     Descargalo desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verificar si existe el modelo entrenado
if not exist "%SCRIPT_DIR%gym_database.pkl" (
    echo.
    echo [!] Advertencia: Base de datos no encontrada
    echo     Se creara automaticamente desde el CSV
    echo.
)

REM Verificar si existe el archivo HTML
if not exist "%SCRIPT_DIR%gym_ui.html" (
    echo.
    echo [X] Error: gym_ui.html no encontrado
    pause
    exit /b 1
)

echo [*] Iniciando servidor web...
echo.
echo ================================================
echo   INSTRUCCIONES:
echo ================================================
echo.
echo   1. El servidor se esta iniciando...
echo   2. Espera a que veas el mensaje de confirmacion
echo   3. Abre tu navegador en:
echo.
echo      http://localhost:5000
echo.
echo   4. Para cerrar, presiona Ctrl+C en esta ventana
echo.
echo ================================================
echo.

REM Iniciar servidor
python "%SCRIPT_DIR%servidor_simple.py"

if errorlevel 1 (
    echo.
    echo [X] Error al iniciar el servidor
    echo.
    pause
)
