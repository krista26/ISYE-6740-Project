# ISYE-6740-Project

Background 

According to GE healthcare, over 90% of healthcare data comes from medical imaging and more than 97% of images captured are not analyzed2. There are not enough radiologists in developing countries to diagnose all images efficiently and accurately. Even in developed nations, waiting times for image analysis can be over 30 days, depending on resources or availability of physicians who can accurately diagnose patients2. Another advantage of using AI for image classification is a higher accuracy of classification when compared to pathologist diagnosis. The Google powered AI for detecting breast cancer metastasis has a 99% accuracy compared to human pathologists, who can miss the diagnosis 62% of the time2. The lack of resources and lower accuracy of a human pathologist can cause a significant delay in the start of treatment. When the treatment is delayed due to lack of a diagnosis, there are significantly higher costs involved with misdiagnosis than there are for over-diagnosis. For example, risks such as higher medical bills, lower life expectancy, and reduction of quality of life3. Having a means to minimize these risks would be extremely beneficial to the communities and overall wellbeing of patients. 

Creating a one-size-fits-all AI model for MRI diagnosis can be challenging because MRI images can come in assorted sizes, different positioning of the head, with varying levels of contrast and background noise. If all images fed into the model are regularized, the accuracy of the model could potentially increase when seeing a fully new data set, thus expanding the scope of the model’s usability. We can utilize suggestions given by high performers of the Large Scale Visual Recognition Challenge to preprocess all images and possibly increase model performance1.  

 

Problem statement 

Identifying patients with brain tumors can be complicated, and with increasing demands on medical providers, there may be missed opportunities to diagnose patients. If diagnosed early enough, managing tumors through lifestyle changes or earlier invasive surgery can lead to a longer, healthier life for patients. Our project is to develop a process for regularizing images and detecting tumors, thus expediting the diagnosis process for a quicker medical response and relieving the resource burden in developing countries. 
