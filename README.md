# virtual_piano_keyboard
"""
Rebecca Magier


To create a camera based piano keyboard that accu- rately responds to the user's input and creates quick sound and highlights the key when detecting if a key is pressed. A brief overview of our algorithm is in- cluded below (Algorithm 1). To improve the usability and user experience we speci􏰃ed the following con- straints:

	User will be able to draw their own desired key- board to interact with
	Use of only one camera
	Ability to work in di􏰂erent environments (light- ing, cameras, skin color, etc)
	Ability to track the keyboard in di􏰂erent posi- tions in the frame so that setup only has to occur once
	Limit false positive key press detection to limit undesirable noise. This is preferable to having to hold a key or press it twice to get a sound

