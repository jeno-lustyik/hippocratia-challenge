import joblib

model_joblib = joblib.load('model.pkl')

while True:
    age = int(input("How old are you? \n"))
    sex = int(input("What is your gender? \n"))
    cp = int(input("Which type of chest pain do you have? [0, 1, 2, 3] \n"))
    trtbps = int(input("What is your resting blood pressure? \n"))
    chol = int(input("What is your cholesterol level? \n"))
    fbs = int(input("Is your fasting blood sugar higher than 120 mg/dl? \n"))
    restecg = int(input("What are your resting electrocardiographic results? [0,1,2] \n"))
    thalach = int(input("What is your maximum heartrate? \n"))
    exang = int(input("Do you have exercise induced angina? \n"))
    oldpeak = float(input("Your ST depression induced by exercise relative to rest \n"))
    slp = int(input("The slope of your peak exercise segment: [0, 1, 2] \n"))
    caa = int(input("The number of your major vessels: [0-3] \n"))
    thal = int(input("Thal: [1, 2, 3] \n"))
    '''
    Preprocess
    predict

    '''
    x_test = [[age, sex, cp, trtbps, chol, fbs, restecg, thalach, exang, oldpeak, slp, caa, thal]]
    prediction = model_joblib.predict(x_test)
    if prediction == 0:
        print("You're safe!")
    else:
        print("Use your Hippocratia product!")