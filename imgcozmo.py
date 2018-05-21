#!/usr/bin/env python3

import numpy as np

from asyncio import Queue
import cozmo
import time

from imgclassification import ImageClassifier
from skimage import color

async def run(robot: cozmo.robot.Robot):
    '''The run method runs once the Cozmo SDK is connected.'''

    img_clf = ImageClassifier()
    # load images
    (train_raw, train_labels) = img_clf.load_data_from_folder('./train/')
    # convert images into features
    train_data = img_clf.extract_image_features(train_raw)
    # train model and test on training data
    img_clf.train_classifier(train_data, train_labels)

    try:
        last_10 = Queue(maxsize=10)
        counts = {}
        while True:
            # get camera image
            event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage)
            image = color.rgb2gray(np.asarray(event.image))
            # predict image
            predicted = img_clf.predict_labels(np.expand_dims(image, axis=0))[0]

            # update last 10 seen
            if not last_10.empty():
                removed = last_10.get()
                counts[removed] = counts[removed] - 1

            # add last seen
            last_10.put(predicted)
            if predicted in counts:
                counts[predicted] = counts[predicted] + 1
                # predicted was guessed 8 out of last 10 times
                if counts[predicted] >= 8:
                    if predicted == 'plane':
                        await robot.say_text(predicted).wait_for_completed()
                        await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabDog).wait_for_completed()
                    elif predicted == 'hands':
                        await robot.say_text(predicted).wait_for_completed()
                        await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabFireTruck).wait_for_completed()
                    elif predicted == 'place':
                        await robot.say_text(predicted).wait_for_completed()
                        await robot.play_anim_trigger(cozmo.anim.Triggers.CodeLabFrustrated).wait_for_completed()
                    # reset last 10
                    last_10 = Queue(maxsize=10)
                    counts = {}
                    time.sleep(5)
            else:
                counts[predicted] = 1


    except KeyboardInterrupt:
        print("")
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)

if __name__ == '__main__':
    cozmo.run_program(run, use_viewer = True, force_viewer_on_top = True)
