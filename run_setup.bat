@echo off
echo Setting up RAG Document Assistant...
echo.

REM Run setup script
python setup.py

echo.
echo Setup complete! Press any key to run tests...
pause > nul

REM Run tests
python test_setup.py

echo.
echo Press any key to exit...
pause > nul
