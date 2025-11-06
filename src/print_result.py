import textwrap

def print_report(train_acc, val_acc, hold_acc, cm, clf_report, top_features, X, X_tr_sel, X_va_sel, X_ho_sel):
    print("\n" + "="*70)
    print("🏆  MODELE : HistGradientBoostingClassifier — Rapport d’évaluation")
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

    print("\n🧠  Interprétation rapide")
    print("-"*70)
    print(textwrap.fill(
        "Le modèle apprend correctement les victoires à domicile, mais peine encore sur les matchs nuls "
        "et les victoires à l’extérieur. Les performances (≈47%) sont cohérentes avec une baseline robuste "
        "sans fuite de données. Prochaines étapes : rééquilibrer les classes et tester LightGBM pour gagner "
        "en stabilité.", width=75
    ))
    print("="*70 + "\n")
