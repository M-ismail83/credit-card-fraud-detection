from preprocessing import Preprocessing
from train_model import Trainer
from evaluation import Evaluator 

def main():
    # 1: Preprocessing
    print("1. Prepping data...")
    prep = Preprocessing()
    prep.load_data()
    prep.splitting()
    
    print(f"   Training set size: {prep.X_train.shape}")

    # 2: Model Training and Guess
    print("\n2. Training the model...")
    # Contamination about %2 
    trainer = Trainer(contamination=0.02) 
    
    # Train model with only x train
    trainer.train(prep.X_train)
    
    print("3. Guessing...")
    y_pred = trainer.predict(prep.X_test)

    # 3: Evaluation
    print("\n4. Evaluating results...")
    eval_obj = Evaluator()
    
    # print report
    eval_obj.print_report(prep.Y_test, y_pred)
        
    #eval_obj.plot_confusion_matrix(prep.Y_test, y_pred)
    
    # 1. Confusion Matrix
    try:
        eval_obj.plot_confusion_matrix(prep.Y_test, y_pred)
    except: pass

    print("calculating the scores...")
    y_scores = trainer.get_scores(prep.X_test)

    # 2. Score dist.
    try:
        eval_obj.plot_score_distribution(prep.Y_test, y_scores)
    except: pass
    
    # 3. PR Curve
    try:
        eval_obj.plot_pr_curve(prep.Y_test, y_scores)
    except: pass

    # 4. t-SNE (optional, slow)
    print("t-SNE visualising (slow :/ )...")
    try:
        eval_obj.plot_tsne(prep.X_test, prep.Y_test, y_pred)
    except: pass

if __name__ == "__main__":
    main()