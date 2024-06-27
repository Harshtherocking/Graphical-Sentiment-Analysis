import os
from sklearn.preprocessing import OneHotEncoder
import joblib

def fix (d : str):
    d = d.removeprefix("\n")
    d = d.removesuffix("\n")
    return [d.strip().lower()]

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), "dep.txt")
    
    with open(path,"r") as file :
        deps = file.readlines()
    file.close()

    deps = map(fix,deps)
    deps = list(deps)

    encoder=OneHotEncoder()
    encoder.fit(deps)

    # x = encoder.transform([deps[8]]).toarray()
    # y = encoder.inverse_transform(x)
    # print(x)
    # print(y)

    with open(os.path.join(os.getcwd(),"dep-encoder.bin"), "wb") as file : 
        joblib.dump(encoder,file)
    file.close()
