import textwrap

def print_report(train_acc, val_acc, hold_acc, cm, clf_report, top_features, X, X_tr_sel, X_va_sel, X_ho_sel):
    print("\n" + "="*70)
    print("🏆 Rapport d’évaluation")
    print("="*70)

    # Résumé global
    print(f"\n📊  Performances globales")
    print("-"*70)
    print(f"  🏋️‍♀️ Train accuracy     : {train_acc:.4f}")
    print(f"  🧪 Validation accuracy : {val_acc:.4f}")
    print(f"  🧊 Hold-out accuracy   : {hold_acc:.4f}")
    print(f"  🧮 Features utilisées  : {X_tr_sel.shape[1]} / {X.shape[1]}")
    print(f"  📚 Échantillons        : train={X_tr_sel.shape[0]} | valid={X_va_sel.shape[0]} | holdout={X_ho_sel.shape[0]}")

    # Matrice de confusion
    print("\n🧩  Matrice de confusion (Hold-out)")
    print("-"*70)
    print(cm)

    # Rapport de classification (formaté)
    print("\n📈  Rapport de classification (Hold-out)")
    print("-"*70)
    print(textwrap.indent(clf_report, "  "))

    # Features importantes
    print("\n🔥  Top 10 features les plus importantes")
    print("-"*70)
    for i, feat in enumerate(top_features[:10], 1):
        print(f"  {i:>2}. {feat}")

    print("="*70 + "\n")
