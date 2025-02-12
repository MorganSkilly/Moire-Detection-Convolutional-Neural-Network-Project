import modeltrainer
import parameters

if __name__ == "__main__":  

    hyperparameters = parameters.Hyperparameters()
    directories = parameters.Directories()

    user_input = input("Please enter 'train' or 'query': ").strip().lower()

    if user_input == "train":        

        modeltrainer.train(hyperparameters, directories)

    elif user_input == "query":

        print("'query'")

    else:

        print("Invalid input. Please enter either 'train' or 'query'.")
