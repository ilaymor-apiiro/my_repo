pipeline {
    agent any

    environment {
        DOCKER_CREDENTIALS_ID = 'docker-credentials' // ID for Docker credentials in Jenkins
        AZURE_CREDENTIALS_ID = 'azure-credentials' // ID for Azure credentials in Jenkins
        IMAGE_REPOSITORY = 'ilaymorapiirostorymaker'
        CONTAINER_REGISTRY = 'aidevelopmentregistry.azurecr.io'
        TAG = "latest"
    }

    stages {
        stage('Initialize') {
            steps {
                script {
                    def dockerHome = tool 'myDocker'
                    if (dockerHome) {
                        env.PATH = "${dockerHome}/bin:${env.PATH}"
                    } else {
                        error("Docker tool not configured")
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    try {
                        docker.build("$CONTAINER_REGISTRY/$IMAGE_REPOSITORY:$TAG", '-f dockerfile .')
                    } catch (Exception e) {
                        error("Docker build failed: ${e.message}")
                    }
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    try {
                        docker.withRegistry("https://$CONTAINER_REGISTRY", DOCKER_CREDENTIALS_ID) {
                            docker.image("$CONTAINER_REGISTRY/$IMAGE_REPOSITORY:$TAG").push()
                        }
                    } catch (Exception e) {
                        error("Docker push failed: ${e.message}")
                    }
                }
            }
        }

        stage('Deploy to Azure Container Instance') {
            steps {
                script {
                    try {
                        azureWebAppPublish azureCredentialsId: AZURE_CREDENTIALS_ID,
                            resourceGroup: 'actual_resource_group',
                            appName: 'actual_app_name',
                            imageName: "$CONTAINER_REGISTRY/$IMAGE_REPOSITORY:$TAG"
                    } catch (Exception e) {
                        error("Deployment to Azure failed: ${e.message}")
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline execution complete.'
        }
    }
}
