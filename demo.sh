#!/bin/bash
# Quick demo launcher for Gray-Scott Reaction-Diffusion implementations

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Gray-Scott Reaction-Diffusion Demo Launcher             ║"
echo "║  Two implementations with time-travel!                    ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Choose an option:"
echo ""
echo "  1) Native JavaScript (instant, no setup)"
echo "  2) THRML Server (unlimited history, JAX-powered)"
echo "  3) Side-by-side Comparison (both at once!)"
echo "  4) Show info"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Native JavaScript version..."
        echo ""
        echo "✨ Features:"
        echo "   • Zero latency"
        echo "   • Works offline"
        echo "   • 1000 frame time-travel buffer"
        echo "   • Runs at 60 FPS"
        echo ""
        cd native-js
        python3 -m http.server 8765 &
        SERVER_PID=$!
        sleep 2
        echo "🌐 Opening http://localhost:8765"
        echo ""
        open "http://localhost:8765" 2>/dev/null || xdg-open "http://localhost:8765" 2>/dev/null || echo "   Visit: http://localhost:8765"
        echo ""
        echo "Press Ctrl+C to stop server"
        wait $SERVER_PID
        ;;

    2)
        echo ""
        echo "🚀 Launching THRML Server version..."
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

        echo ""
        echo "✨ Features:"
        echo "   • Unlimited time-travel history"
        echo "   • JAX-accelerated (GPU capable)"
        echo "   • Scrub to ANY historical frame"
        echo "   • Runs at ~30 FPS"
        echo ""
        cd thrml-server
        echo "🌐 Server starting on http://localhost:5000"
        echo ""
        echo "Press Ctrl+C to stop server"
        python3 server.py
        ;;

    3)
        echo ""
        echo "🚀 Launching Side-by-Side Comparison..."
        echo ""
        echo "Starting THRML server..."
        cd thrml-server
        python3 server.py > /dev/null 2>&1 &
        THRML_PID=$!
        cd ..

        sleep 3

        echo "Starting comparison page..."
        cd comparison
        python3 -m http.server 8766 &
        COMP_PID=$!

        sleep 2
        echo ""
        echo "✨ Both implementations running!"
        echo "   • Left: Native JS (instant)"
        echo "   • Right: THRML Server (unlimited history)"
        echo ""
        echo "🌐 Opening http://localhost:8766"
        echo ""
        open "http://localhost:8766" 2>/dev/null || xdg-open "http://localhost:8766" 2>/dev/null || echo "   Visit: http://localhost:8766"
        echo ""
        echo "Press Ctrl+C to stop both servers"

        trap "kill $THRML_PID $COMP_PID 2>/dev/null" EXIT
        wait $COMP_PID
        ;;

    4)
        echo ""
        echo "╔═══════════════════════════════════════════════════════════╗"
        echo "║  About Gray-Scott Reaction-Diffusion                     ║"
        echo "╚═══════════════════════════════════════════════════════════╝"
        echo ""
        echo "This project implements the Gray-Scott reaction-diffusion model"
        echo "in two ways:"
        echo ""
        echo "1. Native JavaScript"
        echo "   • Client-side simulation"
        echo "   • Ring buffer (last 1000 frames)"
        echo "   • Zero latency"
        echo "   • File: native-js/index.html"
        echo ""
        echo "2. THRML Server"
        echo "   • Server-side with THRML framework"
        echo "   • Unlimited history"
        echo "   • JAX-accelerated"
        echo "   • Files: thrml-server/"
        echo ""
        echo "Both feature:"
        echo "   • Time-travel timeline scrubber"
        echo "   • Play/pause/step controls"
        echo "   • Playback speed control"
        echo "   • Pattern presets (spots, stripes, spirals, worms)"
        echo "   • Interactive painting"
        echo ""
        echo "Time-Travel Controls:"
        echo "   [◄] Step backward"
        echo "   [◄◄] Play in reverse"
        echo "   [▶] Play/Pause"
        echo "   [▶▶] Play forward"
        echo "   [►] Step forward"
        echo "   Drag timeline to scrub"
        echo ""
        echo "Pattern Presets:"
        echo "   • Spots: F=0.055, k=0.062"
        echo "   • Stripes: F=0.035, k=0.060"
        echo "   • Spirals: F=0.014, k=0.054"
        echo "   • Worms: F=0.039, k=0.058"
        echo ""
        echo "For more info, see README.md"
        echo ""
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
