import drawsvg as draw

image_1 = './images/45-1.png' 
image_2 = './images/46-1.png'  

scale = 0.5

role_colors = {
    "r12": "#00DD00",
    "r13": "#00FFFF",
    "r14": "#FF00FF",
    "r15": "#FFA500"
}

d = draw.Drawing(1080*scale, 1170*scale, origin=(0,0), display_inline=False)

d.append(draw.Image(0,0,1080*scale,1170*scale, image_1))
d.append(draw.Image(0,0,1080*scale,1170*scale, image_2))

def split_string(s):
    #split the string into chunks of 8
    return [s[i:i+8] for i in range(0, len(s), 8)]

def create_kilterboard_gui(holdSequence):
    # Create the main window
    holds = split_string(holdSequence)

    # Iterate through the dictionary and draw the holds
    for hold in holds:
        position = hold[0:5]
        role = hold[5:8]
        color = role_colors[role]
        coordinates = get_coords[position]
        x = coordinates[0]
        y = coordinates[1]
        d.append(draw.Circle(x*scale, y*scale, 30*scale, stroke_width=6*scale, fill_opacity=0, stroke=color))


# Save SVG to file
output_file = 'output.svg'
d.save_svg(output_file)
print(f"SVG saved to {output_file}")

# Optional: Save PNG (requires CairoSVG)
try:
    d.save_png('output.png')
    print("PNG version saved to output.png")
except Exception as e:
    print("PNG export skipped:", e)