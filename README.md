<h1 align="center">Hand gesture recognition</h1>

<p align="center">
    <strong>The Hand Gesture Recognition project leverages computer vision and deep learning for efficient gesture detection. It uses OpenCV (cv2) for camera integration and image preprocessing, along with MediaPipe for accurate hand landmark detection. A TensorFlow-based multilayer perceptron (MLP) model is employed to classify gestures, enabling real-time interaction and control.</strong>
    <br />
    <br />
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#contact">Contact</a> •
</p>

<hr />

<h2 id="installation">📁<ins>Installation</ins></h2>
<ul>
    <li>Step 1 : Clone the repo.
    <pre><code>git clone https://github.com/golderalex6/Hand-gesture-recognition.git</code></pre>
    </li>
    <li>Step 2 : Install dependencies.
    <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Step 3 : Setup folders , json files.
    <pre><code>python setup.py</code></pre>
    </li>
</ul>

<h2 id="usage">📈<ins>Usage</ins></h2>
<ul>
    <li>
        <b><ins>Gather your own hand-gesture data : </ins></b><br>
        <ol>
            <li>Run the code below.</li>
            <li>When the OpenCV window opens, position your entire left/right(🖐️)hand in front of the camera(📷).</li>
            <li>Once the hand landmarks or bounding box(🔲) are visible, form the gesture you want(✌️) to capture.</li>
            <li>Keep the gesture steady, but slightly tilt your hand left or right or make small vibrations to generalize the data.</li>
        </ol>
<pre><code>from hand_landmark_data import hand_landmark_data</br>
#Start recording the 'hi' action with the right hand for 300 frames
hand_landmark=hand_landmark_data()
hand_landmark.record('hi','right',300)
</code></pre>
    <i>After collecting the data, the code will generate an 'encode/encode.json' file and 'data' folder. Avoid modifying the (file/folder)'s content or names to prevent errors !!</i>
    </li>
    <li>
        <b><ins>Trainning and evalutate model : </ins></b>
        <ol>
            <li><b>Model's hyperparameters</b> : all your model'config should be put on 'metadata/model_metadata.json' file.</li>
<pre><code># model_metadata.json example
{
    "layers":[200,100,50,20,10],
    "activation":"sigmoid",
    "loss":"sparse_categorical_crossentropy",
    "optimizer":"sgd",
    "epochs":100,
    "batch_size":32
}
</code></pre>
If the 'model_metadata.json' file is not provided, the class will default to using preset hyperparameters .
            <li><b>Trainning and evalutate : </b>After training, the weights will be saved in the 'model' folder. You can also place your pre-trained weights in this folder, rename them to 'hand_landmark_model.weights.h5,' and ensure they are compatible with the project for it to work correctly.</li>
<pre><code>from hand_landmark_model import hand_landmark_model</br>
landmark_model=hand_landmark_model()
landmark_model.train()
landmark_model.evaluate()
</code></pre>
        </ol>
    </li>
    <li><b><ins>Gather your own hand-gesture data : </ins></b>Run the code below. When the OpenCV window opens, record your gestures, and the model will identify the gesture and determine which hand (left or right) it belongs to.</li>
<pre><code>from hand_landmark_model import hand_landmark_model</br>
landmark_model=hand_landmark_model()
landmark_model.gesture_predict()
</code></pre>
</ul>


<h2 id="features">📜<ins>Features</ins></h2>
<ul>
    <li><b>Custome data : </b>Easily record and train the model with any gesture of your choice.</li>
    <li><b>Real-time gesture detection : </b>Leverage MediaPipe and OpenCV to detect gestures instantly and seamlessly in real time.</li>
</ul>

</ul>
<h2 id="contact">☎️<ins>Contact</ins></h2>
<p>
    Golderalex6 - <a href="mailto:golderalex6@gmail.com">golderalex6@gmail.com</a><br>
    Project Link: <a href="https://github.com/golderalex6/Hand-gesture-recognition.git">https://github.com/golderalex6/Hand-gesture-recognition.git</a>
</p>

<hr/>

