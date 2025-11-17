#!/bin/bash

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Gray-Scott Reaction-Diffusion with THRML                ║"
echo "║   Interactive Time-Travel Simulation                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🧪 THRML Playground - Unlimited History & Time-Travel"
echo ""
echo "Choose an option:"
echo ""
echo "  1) Launch THRML Server"
echo "  2) Show project info"
echo "  3) Exit"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching THRML Server..."
        echo ""

        # Check if dependencies are installed
        if ! python3 -c "import flask" 2>/dev/null; then
            echo "⚠️  Dependencies not installed!"
            echo ""
            read -p "Install now? [y/N]: " install
            if [[ $install =~ ^[Yy]$ ]]; then
                cd thrml-server
                pip install -r requirements.txt
            else
                echo "Please run: cd thrml-server && pip install -r requirements.txt"
                exit 1
            fi
        fi

        echo "✨ Features:"
        echo "   • Unlimited simulation history"
        echo "   • JAX-accelerated computation"
        echo "   • Perfect time-travel through all steps"
        echo "   • Factor graph implementation"
        echo ""
        cd thrml-server
        echo "🌐 Server starting on http://localhost:5001"
        echo ""
        echo "Press Ctrl+C to stop server"
        python3 server.py
        ;;

    2)
        clear
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║            Project Information                             ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        echo "This project demonstrates THRML's capabilities through an"
        echo "interactive Gray-Scott reaction-diffusion simulation."
        echo ""
        echo "📚 What is THRML?"
        echo "   THRML (Thermal) is a probabilistic programming framework"
        echo "   built on JAX that uses factor graphs and Gibbs sampling"
        echo "   to model complex systems."
        echo ""
        echo "🎯 Key Innovation:"
        echo "   THRML's sample_states() function automatically preserves"
        echo "   the complete simulation history, making time-travel a"
        echo "   built-in feature rather than an add-on."
        echo ""
        echo "🔬 Gray-Scott Model:"
        echo "   Simulates two chemical species (U and V) with reaction"
        echo "   and diffusion dynamics, producing fascinating patterns"
        echo "   like spots, stripes, spirals, and worms."
        echo ""
        echo "📁 Files:"
        echo "   • thrml-server/simulation.py  - Factor graph implementation"
        echo "   • thrml-server/server.py      - Flask API server"
        echo "   • thrml-server/static/         - Interactive playground UI"
        echo ""
        echo "📖 Documentation:"
        echo "   • README.md                    - Full documentation"
        echo "   • QUICKSTART.md                - Quick start guide"
        echo "   • IMPLEMENTATION_SUMMARY.md    - Technical details"
        echo ""
        echo "🌐 Usage:"
        echo "   Run option 1 to start the server, then open your browser"
        echo "   to http://localhost:5001 for the interactive playground."
        echo ""
        echo "💡 Try different F and k parameters to create:"
        echo "   • Spots (F=0.055, k=0.062)"
        echo "   • Stripes (F=0.035, k=0.060)"
        echo "   • Spirals (F=0.014, k=0.054)"
        echo "   • Worms (F=0.039, k=0.058)"
        echo ""
        read -p "Press Enter to return to menu..."
        exec "$0"
        ;;

    3)
        echo ""
        echo "👋 Goodbye!"
        echo ""
        exit 0
        ;;

    *)
        echo ""
        echo "❌ Invalid choice. Please enter 1, 2, or 3."
        echo ""
        sleep 2
        exec "$0"
        ;;
esac
