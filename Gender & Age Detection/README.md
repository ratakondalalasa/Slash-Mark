


<h2>Objective :</h2>
<p>To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.</p>

<h2>About the Project :</h2>
<p>In this Python project, Deep Learning was utilized to precisely discern the gender and approximate age of individuals from single facial images. The gender prediction yields either 'Male' or 'Female', while the age prediction falls within predefined ranges: (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). Given challenges such as makeup, lighting variations, obstructions, and facial expressions, pinpointing an exact age from a single image is exceptionally arduous. Hence, the problem was formulated as a classification task rather than regression.</p>

<h2>Dataset :</h2>
<p> For this Python project, I utilized the Adience dataset, which is publicly available and accessible via <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">this link</a>. This dataset serves as a benchmark for face photos, encompassing various real-world imaging conditions such as noise, lighting, pose, and appearance. The images are sourced from Flickr albums and are distributed under the Creative Commons (CC) license. The dataset comprises 26,580 photos of 2,284 subjects across eight age ranges (as previously mentioned) and occupies approximately 1GB of storage space. The models employed in this project were trained using this dataset.</p>

<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>argparse</li>
  
       pip install argparse
</ul>

<h2>The contents of this Project :</h2>
<ul>
  <li>opencv_face_detector.pbtxt</li>
  <li>opencv_face_detector_uint8.pb</li>
  <li>age_deploy.prototxt</li>
  <li>age_net.caffemodel</li>
  <li>gender_deploy.prototxt</li>
  <li>gender_net.caffemodel</li>
  <li>a few pictures to try the project on</li>
  <li>detect.py</li>
 </ul>
 <p> For face detection, we utilize a .pb file, which is a protobuf file (protocol buffer) containing both the graph definition and the trained weights of the model. This file enables us to execute the trained model effectively. While a .pb file stores the protobuf in binary format, a file with the .pbtxt extension contains it in text format. Both are TensorFlow files. Regarding age and gender prediction, the .prototxt files outline the network configuration, while the .caffemodel file specifies the internal states of the parameters within the layers.</p>
 
 <h2>Usage :</h2>
 <ul>
  <li>Download my Repository</li>
  <li>Open your Command Prompt or Terminal and change directory to the folder where all the files are present.</li>
  <li><b>Detecting Gender and Age of face in Image</b> Use Command :</li>
  
      python detect.py --image <image_name>
</ul>
  <p><b>Note: </b>The Image should be present in same folder where all the files are present</p> 
<ul>
  <li><b>Detecting Gender and Age of face through webcam</b> Use Command :</li>
  
      python detect.py
</ul>
<ul>
  <li>Press <b>Ctrl + C</b> to stop the program execution.</li>
</ul>

<h2>Examples :</h2>
<p><b>NOTE:- I downloaded the images from Google,if you have any query or problem i can remove them, i just used it for Educational purpose.</b></p>

    >python detect.py --image girl1.jpg
    Gender: Female
    Age: 25-32 years
    
<img src="Example/Detecting age and gender girl1.png">

    >python detect.py --image girl2.jpg
    Gender: Female
    Age: 8-12 years
    
<img src="Example/Detecting age and gender girl2.png">

    >python detect.py --image kid1.jpg
    Gender: Male
    Age: 4-6 years    
    
<img src="Example/Detecting age and gender kid1.png">

    >python detect.py --image kid2.jpg
    Gender: Female
    Age: 4-6 years  
    
<img src="Example/Detecting age and gender kid2.png">

    >python detect.py --image man1.jpg
    Gender: Male
    Age: 38-43 years
    
<img src="Example/Detecting age and gender man1.png">

    >python detect.py --image man2.jpg
    Gender: Male
    Age: 25-32 years
    
<img src="Example/Detecting age and gender man2.png">

    >python detect.py --image woman1.jpg
    Gender: Female
    Age: 38-43 years
    
<img src="Example/Detecting age and gender woman1.png">
              
