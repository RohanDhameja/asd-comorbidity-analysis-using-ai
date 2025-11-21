#!/bin/bash

# ASD Comorbidity & SDOH Dashboard Launcher
# Author: Rohan Dhameja

echo "================================================"
echo "🧠 ASD Comorbidity & SDOH Interactive Dashboard"
echo "================================================"
echo ""
echo "Starting the dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "⚠️  Streamlit is not installed!"
    echo ""
    echo "Installing required packages..."
    pip install -r requirements.txt
    echo ""
fi

# Run the Streamlit app
echo "✅ Launching dashboard..."
echo ""
echo "📍 The dashboard will open in your browser at: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   - Press Ctrl+C to stop the dashboard"
echo "   - Navigate using the sidebar"
echo "   - All visualizations are interactive!"
echo ""
echo "================================================"
echo ""

streamlit run app.py


