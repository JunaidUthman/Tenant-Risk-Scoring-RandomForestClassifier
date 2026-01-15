@Library('Jenkins-shared-library') _

mlModelPipeline(
    appName: 'tenant-risk-scoring-randomforestclassifier',
    hfRepo: 'saaymo/Tenant-Risk-Scoring-RandomForestClassifier',
    modelFiles: [
        [name: 'tenant_risk_model.pkl', targetDir: 'model/models']
    ]
)
