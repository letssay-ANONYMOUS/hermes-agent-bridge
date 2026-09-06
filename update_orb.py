import re

with open('static/index.html', 'r') as f:
    content = f.read()

# 1. Remove overscroll-behavior
content = re.sub(r'^\s*overscroll-behavior:\s*(none|contain);\n', '', content, flags=re.MULTILINE)

# 2. Add canvas to the orb HTML
content = content.replace(
    '<div class="orb" id="orb">',
    '<div class="orb" id="orb">\n            <canvas id="orbCanvas"></canvas>'
)

# 3. Replace old orb CSS
old_orb_css = """      .orb {
        width: min(62vw, 250px);
        aspect-ratio: 1;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.14);
        position: relative;
        display: grid;
        place-items: center;
        background:
          radial-gradient(circle at 44% 40%, rgba(196,236,215,.28), transparent 22%),
          radial-gradient(circle at 50% 55%, rgba(127,201,163,.18), rgba(255,255,255,.05) 55%, rgba(255,255,255,.02));
        box-shadow: inset 0 1px 0 rgba(255,255,255,.12), inset 0 -20px 60px rgba(0,0,0,.25);
        transition: transform .3s cubic-bezier(.16,1,.3,1), border-color .3s;
      }

      .orb::before, .orb::after {
        content: "";
        position: absolute;
        inset: 14%;
        border-radius: 50%;
        border: 1px solid rgba(127,201,163,.4);
        opacity: .22;
        animation: pulse 2.6s infinite cubic-bezier(.16,1,.3,1);
      }
      .orb::after { inset: 30%; animation-delay: .5s; }

      .orb.listening { transform: scale(1.04); border-color: rgba(127,201,163,.55); }
      .orb.thinking::before, .orb.thinking::after { animation-duration: 1s; border-color: rgba(210,176,111,.55); }
      .orb.speaking::before, .orb.speaking::after { animation-duration: .7s; }
      .orb.interrupted { border-color: rgba(224,140,140,.5); }

      @keyframes pulse {
        0% { transform: scale(.72); opacity: 0; }
        32% { opacity: .55; }
        100% { transform: scale(1.3); opacity: 0; }
      }"""

new_orb_css = """      .orb {
        width: min(62vw, 250px);
        aspect-ratio: 1;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,.14);
        position: relative;
        display: grid;
        place-items: center;
        background: transparent;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.12), inset 0 -20px 60px rgba(0,0,0,.25);
        transition: transform .3s cubic-bezier(.16,1,.3,1), border-color .3s;
        overflow: hidden;
      }
      
      .orb canvas {
        position: absolute;
        inset: -20%;
        width: 140%;
        height: 140%;
        border-radius: 50%;
        mix-blend-mode: screen;
        pointer-events: none;
        z-index: 0;
      }
      
      .orb-label { z-index: 10; position: relative; }
      
      .orb.listening { transform: scale(1.04); border-color: rgba(127,201,163,.55); }
      .orb.thinking { border-color: rgba(210,176,111,.55); }
      .orb.interrupted { border-color: rgba(224,140,140,.5); }"""

content = content.replace(old_orb_css, new_orb_css)

# 4. Inject Orb Canvas JS
orb_js = """
      // ====== Interactive Orb Logic ======
      const canvas = document.getElementById('orbCanvas');
      if (canvas) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        let points = [];
        const numPoints = 80;
        
        // Settings for states
        const states = {
          idle:      { speed: 0.015, amplitude: 3,  color: 'rgba(127, 201, 163, 0.4)', radius: 80, noise: 0.3 },
          listening: { speed: 0.03,  amplitude: 6,  color: 'rgba(100, 220, 180, 0.6)', radius: 85, noise: 0.5 },
          thinking:  { speed: 0.06,  amplitude: 15, color: 'rgba(210, 176, 111, 0.7)', radius: 90, noise: 1.2 },
          speaking:  { speed: 0.05,  amplitude: 12, color: 'rgba(230, 245, 255, 0.8)', radius: 95, noise: 0.8 },
        };
        
        let currentState = states.idle;
        
        function resizeCanvas() {
          canvas.width = canvas.offsetWidth;
          canvas.height = canvas.offsetHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        // Call it initially in a timeout to ensure CSS is applied
        setTimeout(resizeCanvas, 100);
        
        // Helper: soft lerp
        function lerp(start, end, amt) { return (1 - amt) * start + amt * end; }
        
        // Active visual params
        let currentVisuals = { ...states.idle };
        
        function drawOrb() {
          if(!canvas.width) resizeCanvas();
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          // Determine target state based on UI class
          let target = states.idle;
          if (orb.classList.contains('speaking')) target = states.speaking;
          else if (orb.classList.contains('thinking')) target = states.thinking;
          else if (orb.classList.contains('listening')) target = states.listening;
          
          // React to audio level when speaking
          let dynamicAmp = target.amplitude;
          let dynamicRad = target.radius;
          let dynamicNoise = target.noise;
          
          if (target === states.speaking) {
            let lvl = parseFloat(orbMeter.textContent) || 0;
            // Map 0-1 audio level to extra amplitude and radius
            dynamicAmp += lvl * 40;
            dynamicRad += lvl * 20;
            dynamicNoise += lvl * 2;
          }
          
          // Smooth transitions
          currentVisuals.speed = lerp(currentVisuals.speed, target.speed, 0.05);
          currentVisuals.amplitude = lerp(currentVisuals.amplitude, dynamicAmp, 0.1);
          currentVisuals.radius = lerp(currentVisuals.radius, dynamicRad, 0.1);
          currentVisuals.noise = lerp(currentVisuals.noise, dynamicNoise, 0.05);
          
          time += currentVisuals.speed;
          
          const centerX = canvas.width / 2;
          const centerY = canvas.height / 2;
          
          ctx.beginPath();
          for (let i = 0; i <= numPoints; i++) {
            const angle = (i / numPoints) * Math.PI * 2;
            
            // Create some wavy math for amorphous blob
            const noise = 
              Math.sin(angle * 3 + time) * 0.5 + 
              Math.cos(angle * 5 - time * 0.8) * 0.3 + 
              Math.sin(angle * 2 + time * 1.5) * 0.2;
              
            const offset = noise * currentVisuals.amplitude * currentVisuals.noise;
            const r = currentVisuals.radius + offset;
            
            const x = centerX + Math.cos(angle) * r;
            const y = centerY + Math.sin(angle) * r;
            
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.closePath();
          
          // Radial gradient for 3D look
          const gradient = ctx.createRadialGradient(
            centerX - 20, centerY - 20, 0,
            centerX, centerY, currentVisuals.radius + 30
          );
          
          // Interpolate color gently (basic hack for seamless color)
          ctx.fillStyle = target.color; 
          ctx.fill();
          
          // Inner glow
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'rgba(255,255,255,0.4)';
          ctx.stroke();
          
          requestAnimationFrame(drawOrb);
        }
        
        drawOrb();
      }
"""

content = content.replace("</script>", orb_js + "\n    </script>")

with open('static/index.html', 'w') as f:
    f.write(content)

