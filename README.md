# Charades Webcam

A real-time smart webcam application using TensorFlow and [OpenCV](http://opencv.org/).

![Holding a webcam demo](./media/holding_webcam.gif)

![Watching on a screen demo](./media/watching_on_screen.gif)

## Information

The key to building intelligent AI systems is data: Data with the right insight into our lives. Since 2016 we have been using our [Charades dataset](http://allenai.org/plato/charades/) to train models that understand videos of boring daily activities, such as `watching TV`, `sitting on a couch`, `looking outside a window`. This repository takes a tiny/fast SqueezeNet 1.1 frozen tensorflow model trained on the Charades dataset and runs it on a real-time webcam feed. Note that this is a simple model that achieves 13.5% mean average precision on the Charades benchmark, but there are now sophisticated models that obtain 34.4% (Google DeepMind) and 39.5% (Carnegie Mellon University), therefore this is a simple real-time model that we hope will allow everyone to use this in various applications.

## Getting Started
1. Packages:  matplotlib, numpy via pip, or alternatively `conda env create -f environment.yml`
2. `python charades_webcam.py`
    Optional arguments (default value):
    * Device index of the camera `--source=0`
    * Width of the frames in the video stream `--width=480`
    * Height of the frames in the video stream `--height=360`
    * Number of workers `--num-workers=2`
    * Size of the queue `--queue-size=5`

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)

## Notes
- OpenCV 3.1 might crash on OSX after a while. See open issue and solution [here](https://github.com/opencv/opencv/issues/5874).

## Acknowledgements 
Shoutout to Dat Tran for a great real-time object detector that was the basis for this code.
https://github.com/datitran/object_detector_app

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2018 [Gunnar Sigurdsson](https://github.com/gsig).
