# ============================================================
# GYM AI DEPLOYMENT TOOL - PowerShell Version
# Versión: 1.0
# Windows 10/11 Compatible
# ============================================================

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("python", "flutter", "update", "clean", "logs")]
    [string]$Mode = "menu"
)

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ScriptDir "gym_ai_deploy.log"
$VenvPath = Join-Path $ScriptDir "venv"
$FlutterDir = Join-Path $ScriptDir "flutter"
$ConfigFile = Join-Path $ScriptDir "config.json"
$PubspecFile = Join-Path $ScriptDir "pubspec.yaml"

# Colores para la consola
$ColorError = 'Red'
$ColorSuccess = 'Green'
$ColorWarning = 'Yellow'
$ColorInfo = 'Cyan'

# ============================================================
# FUNCIONES DE UTILIDAD
# ============================================================

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    
    Add-Content -Path $LogFile -Value $logMessage
    Write-Host $logMessage
}

function Write-Success {
    param([string]$Message)
    Write-Host "[✓] $Message" -ForegroundColor $ColorSuccess
    Write-Log $Message
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $ColorError
    Write-Log "ERROR: $Message"
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor $ColorWarning
    Write-Log "WARNING: $Message"
}

function Write-Info {
    param([string]$Message)
    Write-Host "[*] $Message" -ForegroundColor $ColorInfo
    Write-Log $Message
}

function Test-InternetConnection {
    Write-Info "Verificando conexión a Internet..."
    try {
        $response = Invoke-WebRequest -Uri "https://www.google.com" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        Write-Success "Conexión a Internet OK"
        return $true
    } catch {
        Write-Error-Custom "No hay conexión a Internet"
        return $false
    }
}

# ============================================================
# FUNCIONES PYTHON
# ============================================================

function Test-Python {
    Write-Info "Verificando Python..."
    
    try {
        $version = python --version 2>&1
        Write-Success "Python encontrado: $version"
        return $true
    } catch {
        Write-Warning-Custom "Python no encontrado"
        return $false
    }
}

function Install-Python {
    Write-Info "Instalando Python 3.11..."
    
    if (-not (Test-InternetConnection)) {
        Write-Error-Custom "Se requiere conexión a Internet para descargar Python"
        return $false
    }
    
    # Intentar con winget (Windows 11)
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Info "Usando winget para instalar Python..."
        try {
            & winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements --silent
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Python instalado con winget"
                return $true
            }
        } catch {
            Write-Warning-Custom "winget falló, intentando otra método..."
        }
    }
    
    # Intentar con Chocolatey
    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($choco) {
        Write-Info "Usando Chocolatey para instalar Python..."
        try {
            & choco install python -y --silent
            if ($LASTEXITCODE -eq 0) {
                Write-Success "Python instalado con Chocolatey"
                return $true
            }
        } catch {
            Write-Warning-Custom "Chocolatey falló, intentando descarga directa..."
        }
    }
    
    # Descarga manual
    return Download-Python
}

function Download-Python {
    Write-Info "Descargando instalador Python (esto puede tomar unos minutos)..."
    
    $PythonURL = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
    $PythonInstaller = Join-Path $ScriptDir "python_installer.exe"
    
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        (New-Object Net.WebClient).DownloadFile($PythonURL, $PythonInstaller)
        
        Write-Info "Ejecutando instalador Python..."
        & $PythonInstaller /quiet InstallAllUsers=1 PrependPath=1
        
        Start-Sleep -Seconds 5
        Remove-Item $PythonInstaller -Force -ErrorAction SilentlyContinue
        
        Write-Success "Python instalado exitosamente"
        return $true
    } catch {
        Write-Error-Custom "Error descargando Python: $_"
        return $false
    }
}

function Setup-Python {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "  SETUP PYTHON Y BACKEND (Opción 1)" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    
    Write-Log "--- Iniciando Setup Python ---"
    
    # Verificar Python
    if (-not (Test-Python)) {
        if (-not (Install-Python)) {
            Write-Error-Custom "Python no se pudo instalar"
            Read-Host "Presiona Enter para volver al menú"
            return $false
        }
    }
    
    # Crear entorno virtual
    if (-not (Test-Path $VenvPath)) {
        Write-Info "Creando entorno virtual..."
        Write-Log "Creando entorno virtual en: $VenvPath"
        
        try {
            & python -m venv $VenvPath
            Write-Success "Entorno virtual creado"
            Write-Log "Entorno virtual creado exitosamente"
        } catch {
            Write-Error-Custom "No se pudo crear el entorno virtual: $_"
            return $false
        }
    } else {
        Write-Success "Entorno virtual ya existe"
    }
    
    # Activar entorno virtual
    $activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
    & $activateScript
    Write-Log "Entorno virtual activado"
    
    # Instalar dependencias
    Write-Host ""
    Write-Info "Instalando dependencias de Python..."
    Write-Log "Instalando dependencias Python"
    
    try {
        Write-Info "Actualizando pip..."
        & python -m pip install --upgrade pip --quiet
        
        Write-Info "Instalando paquetes: torch, transformers, pandas, scikit-learn..."
        & python -m pip install torch transformers pandas scikit-learn numpy --quiet
        
        Write-Success "Dependencias instaladas exitosamente"
        Write-Log "Dependencias instaladas: torch, transformers, pandas, scikit-learn, numpy"
    } catch {
        Write-Error-Custom "Error instalando dependencias: $_"
        return $false
    }
    
    # Verificar que existe nngym_v2.py
    $pyScript = Join-Path $ScriptDir "nngym_v2.py"
    if (-not (Test-Path $pyScript)) {
        Write-Error-Custom "No se encuentra nngym_v2.py"
        return $false
    }
    
    # Ejecutar script
    Write-Host ""
    Write-Info "Ejecutando nngym_v2.py..."
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    Write-Log "Ejecutando script principal: nngym_v2.py"
    
    try {
        & python $pyScript
        Write-Success "Ejecución completada"
        Write-Log "Script completado"
    } catch {
        Write-Error-Custom "Error ejecutando script: $_"
        return $false
    }
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Read-Host "Presiona Enter para volver al menú"
    return $true
}

# ============================================================
# FUNCIONES FLUTTER
# ============================================================

function Test-Flutter {
    Write-Info "Verificando Flutter..."
    
    # Verificar en PATH
    $flutter = Get-Command flutter -ErrorAction SilentlyContinue
    if ($flutter) {
        $version = & flutter --version 2>&1
        Write-Success "Flutter encontrado: $version"
        return $true
    }
    
    # Verificar en directorio local
    $flutterExe = Join-Path $FlutterDir "flutter\bin\flutter.bat"
    if (Test-Path $flutterExe) {
        Write-Success "Flutter encontrado en directorio local"
        # Agregar al PATH temporalmente
        $env:PATH = "$(Join-Path $FlutterDir 'flutter\bin');$env:PATH"
        return $true
    }
    
    Write-Warning-Custom "Flutter no encontrado"
    return $false
}

function Install-Flutter {
    Write-Info "Instalando Flutter..."
    Write-Log "Instalando Flutter"
    
    if (-not (Test-InternetConnection)) {
        Write-Error-Custom "Se requiere conexión a Internet para descargar Flutter"
        return $false
    }
    
    if (-not (Test-Path $FlutterDir)) {
        New-Item -ItemType Directory -Path $FlutterDir | Out-Null
    }
    
    Write-Info "Descargando Flutter (esto puede tomar unos minutos)..."
    
    $FlutterURL = "https://storage.googleapis.com/flutter_infra_release/releases/stable/windows/flutter_windows_3.16.0-stable.zip"
    $FlutterZip = Join-Path $ScriptDir "flutter_installer.zip"
    
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        (New-Object Net.WebClient).DownloadFile($FlutterURL, $FlutterZip)
        
        Write-Info "Extrayendo Flutter..."
        Expand-Archive -Path $FlutterZip -DestinationPath $FlutterDir -Force
        
        Remove-Item $FlutterZip -Force
        
        # Agregar al PATH
        $env:PATH = "$(Join-Path $FlutterDir 'flutter\bin');$env:PATH"
        
        Write-Success "Flutter instalado en: $FlutterDir"
        Write-Log "Flutter instalado en: $FlutterDir"
        
        return $true
    } catch {
        Write-Error-Custom "Error descargando Flutter: $_"
        return $false
    }
}

function Create-Flutter-Project {
    Write-Info "Creando estructura del proyecto Flutter..."
    Write-Log "Creando estructura Flutter"
    
    # Crear directorios
    $libDir = Join-Path $ScriptDir "lib"
    $assetsDir = Join-Path $ScriptDir "assets"
    
    if (-not (Test-Path $libDir)) {
        New-Item -ItemType Directory -Path $libDir | Out-Null
    }
    if (-not (Test-Path $assetsDir)) {
        New-Item -ItemType Directory -Path $assetsDir | Out-Null
    }
    
    # Crear main.dart (versión simplificada para PowerShell)
    $mainDartContent = @'
import 'package:flutter/material.dart';

void main() {
  runApp(const GymAIApp());
}

class GymAIApp extends StatelessWidget {
  const GymAIApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'GYM AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primaryColor: const Color(0xFFFF0000),
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFFF0000),
        ),
      ),
      home: const GymAIHomePage(),
    );
  }
}

class GymAIHomePage extends StatefulWidget {
  const GymAIHomePage({Key? key}) : super(key: key);

  @override
  State<GymAIHomePage> createState() => _GymAIHomePageState();
}

class _GymAIHomePageState extends State<GymAIHomePage> {
  final TextEditingController _searchController = TextEditingController();
  List<Map<String, dynamic>> _searchResults = [];
  List<String> _searchHistory = [];

  void _performSearch(String query) {
    if (query.isNotEmpty) {
      setState(() {
        if (!_searchHistory.contains(query)) {
          _searchHistory.insert(0, query);
          if (_searchHistory.length > 10) {
            _searchHistory.removeLast();
          }
        }
        _searchResults = [
          {
            'name': 'Ejercicio: $query',
            'score': 0.92,
            'description': 'Descripción del ejercicio encontrado'
          },
        ];
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('GYM AI FINDER'),
        backgroundColor: const Color(0xFF8B0000),
        centerTitle: true,
        elevation: 10,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
              const SizedBox(height: 20),
              TextField(
                controller: _searchController,
                decoration: InputDecoration(
                  hintText: 'Describe el ejercicio...',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                  prefixIcon: const Icon(Icons.fitness_center),
                  prefixIconColor: const Color(0xFFB22222),
                ),
                onSubmitted: (value) {
                  _performSearch(value);
                },
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  _performSearch(_searchController.text);
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFFF0000),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 50,
                    vertical: 20,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                ),
                child: const Text(
                  'BUSCAR EJERCICIO',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 30),
              if (_searchResults.isNotEmpty)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Resultados:',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 15),
                    ..._searchResults.map((result) => Card(
                      elevation: 5,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Container(
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: const Color(0xFFB22222),
                            width: 2,
                          ),
                        ),
                        padding: const EdgeInsets.all(15),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              result['name']!,
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Score: ${result['score']}',
                              style: const TextStyle(
                                color: Color(0xFFFF0000),
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              result['description']!,
                              style: const TextStyle(color: Colors.grey),
                            ),
                          ],
                        ),
                      ),
                    )).toList(),
                  ],
                ),
              if (_searchHistory.isNotEmpty)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 30),
                    const Text(
                      'Historial de búsquedas:',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 8,
                      children: _searchHistory.map((history) => Chip(
                        label: Text(history),
                        backgroundColor: const Color(0xFFB22222),
                        labelStyle: const TextStyle(color: Colors.white),
                        onDeleted: () {
                          setState(() => _searchHistory.remove(history));
                        },
                      )).toList(),
                    ),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }
}
'@

    $mainDartPath = Join-Path $libDir "main.dart"
    Set-Content -Path $mainDartPath -Value $mainDartContent
    Write-Success "main.dart creado"
    Write-Log "main.dart creado"
}

function Setup-Flutter {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "  SETUP FLUTTER Y INTERFAZ (Opción 2)" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    
    Write-Log "--- Iniciando Setup Flutter ---"
    
    # Verificar Flutter
    if (-not (Test-Flutter)) {
        if (-not (Install-Flutter)) {
            Write-Error-Custom "Flutter no se pudo instalar"
            Read-Host "Presiona Enter para volver al menú"
            return $false
        }
    }
    
    # Crear proyecto si no existe
    $mainDart = Join-Path $ScriptDir "lib\main.dart"
    if (-not (Test-Path $mainDart)) {
        Create-Flutter-Project
    }
    
    # Obtener dependencias
    Write-Host ""
    Write-Info "Obteniendo dependencias de Flutter..."
    Write-Log "Descargando dependencias Flutter"
    
    try {
        Push-Location $ScriptDir
        & flutter pub get --quiet
        Pop-Location
    } catch {
        Write-Warning-Custom "Algunos problemas con las dependencias"
        Write-Log "ADVERTENCIA: Problemas con dependencias de Flutter"
    }
    
    # Habilitar modo desktop
    Write-Info "Configurando Flutter para Windows Desktop..."
    & flutter config --enable-windows-desktop --quiet
    Write-Log "Flutter Desktop habilitado"
    
    # Ejecutar flutter
    Write-Host ""
    Write-Info "Ejecutando aplicación Flutter..."
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    Write-Log "Iniciando flutter run"
    
    try {
        Push-Location $ScriptDir
        & flutter run -d windows
        Pop-Location
        Write-Success "Flutter finalizado"
    } catch {
        Write-Error-Custom "Error ejecutando Flutter: $_"
        return $false
    }
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Read-Host "Presiona Enter para volver al menú"
    return $true
}

# ============================================================
# FUNCIONES ADICIONALES
# ============================================================

function Update-All {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "  ACTUALIZAR DEPENDENCIAS (Opción 3)" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    
    Write-Log "--- Actualizando todas las dependencias ---"
    
    Write-Info "Actualizando Python y pip..."
    try {
        & python -m pip install --upgrade pip --quiet
        Write-Info "Actualizando paquetes Python..."
        & python -m pip install --upgrade torch transformers pandas scikit-learn numpy --quiet
        Write-Success "Python actualizado"
    } catch {
        Write-Warning-Custom "Error actualizando Python"
    }
    
    Write-Host ""
    Write-Info "Actualizando Flutter..."
    try {
        & flutter upgrade --quiet
        Write-Success "Flutter actualizado"
    } catch {
        Write-Warning-Custom "Error actualizando Flutter"
    }
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Success "Todas las dependencias actualizadas"
    Write-Log "Actualización completada"
    Write-Host ""
    Read-Host "Presiona Enter para volver al menú"
}

function View-Logs {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "  REGISTRO DE DESPLIEGUE (Opción 4)" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    
    if (Test-Path $LogFile) {
        Get-Content $LogFile
    } else {
        Write-Warning-Custom "Archivo de log no encontrado"
    }
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    Read-Host "Presiona Enter para volver al menú"
}

function Cleanup-Files {
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Red
    Write-Host "  LIMPIAR ARCHIVOS (Opción 5)" -ForegroundColor Red
    Write-Host "======================================" -ForegroundColor Red
    Write-Host ""
    
    Write-Warning-Custom "Se eliminarán:"
    Write-Host "  - Entorno virtual Python ($VenvPath)"
    Write-Host "  - Caché de Flutter"
    Write-Host "  - Archivos temporales"
    Write-Host ""
    
    $confirm = Read-Host "Confirma? (S/N)"
    
    if ($confirm -eq "S" -or $confirm -eq "s") {
        Write-Info "Limpiando archivos..."
        Write-Log "Limpieza de archivos iniciada"
        
        if (Test-Path $VenvPath) {
            Write-Info "Eliminando entorno virtual..."
            Remove-Item -Recurse -Force $VenvPath
            Write-Success "Entorno virtual eliminado"
            Write-Log "Entorno virtual eliminado"
        }
        
        if (Test-Path ".flutter") {
            Remove-Item -Recurse -Force ".flutter"
            Write-Log "Caché Flutter eliminado"
        }
        
        Write-Success "Limpieza completada"
        Write-Log "Limpieza completada"
    } else {
        Write-Info "Limpieza cancelada por el usuario"
        Write-Log "Limpieza cancelada"
    }
    
    Write-Host ""
    Read-Host "Presiona Enter para volver al menú"
}

# ============================================================
# MENÚ PRINCIPAL
# ============================================================

function Show-Menu {
    Write-Host ""
    Write-Host "     ========================================" -ForegroundColor Red
    Write-Host "     |  GYM AI DEPLOYMENT MENU           |" -ForegroundColor Red
    Write-Host "     ========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "     1. [Python] Compilar y Ejecutar Backend" -ForegroundColor Cyan
    Write-Host "     2. [Flutter] Ejecutar Interfaz Gráfica" -ForegroundColor Cyan
    Write-Host "     3. [Actualizar] Actualizar dependencias" -ForegroundColor Cyan
    Write-Host "     4. [Logs] Ver registro de despliegue" -ForegroundColor Cyan
    Write-Host "     5. [Limpiar] Limpiar archivos temporales" -ForegroundColor Cyan
    Write-Host "     6. [Salir]" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "     ========================================" -ForegroundColor Red
    Write-Host ""
}

# Script Principal
Clear-Host
Write-Host ""
Write-Host "██████╗ ███████╗ █████╗ ██╗   ██╗" -ForegroundColor Red
Write-Host "██╔════╝ ██╔════╝██╔══██╗╚██╗ ██╔╝" -ForegroundColor Red
Write-Host "██║  ███╗█████╗  ███████║ ╚████╔╝ " -ForegroundColor Red
Write-Host "██║   ██║██╔══╝  ██╔══██║  ╚██╔╝  " -ForegroundColor Red
Write-Host "╚██████╔╝███████╗██║  ██║   ██║   " -ForegroundColor Red
Write-Host " ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   " -ForegroundColor Red
Write-Host "         AI DEPLOYMENT TOOL" -ForegroundColor Red
Write-Host ""

Write-Log "========== GYM AI DEPLOYMENT INICIADO =========="
Write-Log "Directorio: $ScriptDir"
Write-Log "Hora: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

do {
    Show-Menu
    $choice = Read-Host "  Seleccione opción [1-6]"
    
    switch ($choice) {
        "1" { Setup-Python }
        "2" { Setup-Flutter }
        "3" { Update-All }
        "4" { View-Logs }
        "5" { Cleanup-Files }
        "6" { 
            Write-Host "¡Hasta luego!" -ForegroundColor Green
            exit 0 
        }
        default { 
            Write-Error-Custom "Opción no válida"
            Start-Sleep -Seconds 2
        }
    }
    
    Clear-Host
} while ($true)
