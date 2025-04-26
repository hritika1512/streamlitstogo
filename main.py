import streamlit as st
import random
import os
import requests
import streamlit.components.v1 as components
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

st.set_page_config(page_title="SynapseAI", page_icon="logo.ico", layout="wide")



def mandela_component(color, brush_size, symmetry_lines):
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/konva@8/konva.min.js"></script>
        <style>
            #container {{
                width: 500px !important;
                height: 500px !important;
                border: 4px solid black !important; /* Force a border to see the container */
            }}
            canvas {{
                border: 1px solid black !important; /* Force a border on the canvas */
            }}
            body {{
                margin: 0 !important; /* Reset body margin in iframe */
                overflow: hidden !important; /* Prevent scrollbars in iframe */
            }}
        </style>
    </head>
    <body>
        <div id="container"></div>
        <button id="clearButton">Clear</button>
        <script>
            const stage = new Konva.Stage({{
                container: 'container',
                width: 500,
                height: 500,
            }});
            const layer = new Konva.Layer();
            stage.add(layer);

            const background = new Konva.Rect({{
                x: 0,
                y: 0,
                width: stage.width(),
                height: stage.height(),
                fill: 'white',
                listening: false,
            }});
            layer.add(background);

            let isDrawing = false;
            let strokeColor = '{color}';
            let strokeWidth = '{brush_size}';
            let symmetryLines = parseInt('{symmetry_lines}', 10);
            let lastDrawTime = 0;
            let currentLine;

            stage.on('mousedown touchstart', (e) => {{
                isDrawing = true;
                const pos = stage.getPointerPosition();
                const newLine = new Konva.Line({{
                    points: [pos.x, pos.y],
                    stroke: strokeColor,
                    strokeWidth: strokeWidth,
                    lineCap: 'round',
                    lineJoin: 'round',
                    name: 'userLine'
                }});
                layer.add(newLine);
                currentLine = newLine;
            }});

            stage.on('mousemove touchmove', (e) => {{
                if (!isDrawing) return;
                const currentTime = Date.now();
                if (currentTime - lastDrawTime < 16) return;
                lastDrawTime = currentTime;

                const pos = stage.getPointerPosition();
                const newPoints = currentLine.points().concat([pos.x, pos.y]);
                currentLine.points(newPoints);

                const centerX = stage.width() / 2;
                const centerY = stage.height() / 2;
                const angle = (2 * Math.PI) / symmetryLines;

                layer.getChildren((node) => node.name() === 'symmetryLine' && node.userLineRef === currentLine).forEach((node) => node.destroy());

                for (let i = 1; i < symmetryLines; i++) {{
                    const rotatedPoints = [];
                    for (let j = 0; j < newPoints.length; j += 2) {{
                        const dx = newPoints[j] - centerX;
                        const dy = newPoints[j + 1] - centerY;
                        const rotatedX = dx * Math.cos(angle * i) - dy * Math.sin(angle * i) + centerX;
                        const rotatedY = dx * Math.sin(angle * i) + dy * Math.cos(angle * i) + centerY;
                        rotatedPoints.push(rotatedX, rotatedY);
                    }}
                    const symmetryLine = new Konva.Line({{
                        points: rotatedPoints,
                        stroke: strokeColor,
                        strokeWidth: strokeWidth,
                        lineCap: 'round',
                        lineJoin: 'round',
                        name: 'symmetryLine',
                        userLineRef: currentLine,
                    }});
                    layer.add(symmetryLine);
                }}
                layer.batchDraw();
            }});

            stage.on('mouseup touchend', () => {{
                isDrawing = false;
            }});

            document.getElementById('clearButton').addEventListener('click', function() {{
                layer.getChildren((node) => node.name() === 'userLine' || node.name() === 'symmetryLine').forEach((node) => node.destroy());
                layer.draw();
            }});

            window.addEventListener('message', function(event) {{
                if (event.data.type === 'color_update') {{
                    strokeColor = event.data.color;
                    console.log("Color updated to: ", strokeColor);
                }}
            }});
            console.log("Initial color: ", strokeColor);

        </script>
    </body>
    </html>
    """
    components.html(html_string, width=600, height=600)

st.title("Mandala Drawing Debug")

color = st.color_picker("Choose Color", "#000000")
brush_size = st.slider("Brush Size", 1, 10, 2)
symmetry_lines = st.slider("Symmetry Lines", 2, 20, 8)

mandela_component(color, brush_size, symmetry_lines)

if st.session_state.get('color') != color:
    components.html(f"""
    <script>
        window.dispatchEvent(new MessageEvent('message', {{data: {{type: 'color_update', color: '{color}'}}}}));
        console.log("message dispatched to change color to: ", '{color}');
    </script>
    """, height=0)
    st.session_state['color'] = color
    """, height=0)
    st.session_state['color'] = color
