**🌱 Plant AI: Your Pocket Botanist**
Ever wondered what that leafy friend on your windowsill is actually called and how to keep it happy? Plant AI uses Deep Learning to bridge the gap between curiosity and expert plant care.

**🎯 The Vision**
Can machine learning models identify plants as effectively as humans? We built this application to find out. By analysing visual features through a fine-tuned neural network, Plant AI predicts plant species and provides essential care tips to help your green friends thrive.

**🛠️ Tech Stack**
•	Deep Learning: TensorFlow & Keras
•	Architecture: EfficientNetB2 (Transfer Learning)
•	Optimization: Adam- EarlyStopping & Data Augmentation (to combat overfitting)
•	Testing Suit/ Web Framework: Streamlit
•	Visualization: Matplotlib & Seaborn

**📊 Model Performance & Insights**
The model was evaluated using a confusion matrix, showing strong generalization across most species.
  ✅ The Top Performers: 
  The model achieved near-perfect accuracy for several distinct species:
    •	African Violet: 100% (20/20)
    •	Bird of Paradise: 100% (30/30)
    •	Monstera Deliciosa: 100% (47/47)
    •	Snake Plant: 100% (39/39)
    •	Areca Palm: 100% (40/40)

**🔬 Lessons Learned (Room for Improvement)**
No model is perfect! Our analysis revealed interesting visual "blind spots":
  •	The "Lily of the Valley" Challenge: The model struggled with this species, likely due to a lack of diverse training data.
  •	Visual Mimicry: We observed mutual confusion between Daffodils vs. Hyacinths and Jade Plants vs. Kalanchoe. This mimics human error, as these species share similar leaf structures and textures.
  •	Small Sample Sizes: Rare classes like the Rubber Plant and Tradescantia showed higher variance, highlighting the need for more balanced datasets in future iterations.

Here is the Confusion Matrix for a quick overview. The diagonal elements (where predicted class matches actual class) are generally high compared to the off-diagonal elements. This means that the model is performing well for most of the plant species it's designed to classify. Many classes show a high number of correctly classified instances (True Positives).

![Confusion Matrix]<img width="876" height="764" alt="Image" src="https://github.com/user-attachments/assets/ab2077d2-931e-46ac-bf50-a643fe758495" />


**🚀 How to Run Locally**
  1.	Clone the repo:
      git clone https://github.com/ choudharysujata / Houseplant_Image_Identifier_App.git
  2.	Install dependencies: 
      pip install -r requirements.txt
  3.	Run the Streamlit App
      streamlit run app.py
    	
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
**🌱 The Future Roadmap**
Even the best gardeners learn from their plants! I know where my model fails, and here is how I plan to fix. Currently, the model is a "pro" at identifying Monsteras but gets a bit confused by Lily of the Valleys. My next steps include:
•	Feeding the Model: Gathering more diverse images to help the model "see" the subtle differences between similar-looking flowers.
•	New Species: Expanding the library to include succulents and rare tropical plants.
•	Real-time Care Alerts: Integrating a scheduling system to remind users when to water their newly identified friends.







