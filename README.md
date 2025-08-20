Projet d'Apprentissage Fédéré avec LLaVA
Ce projet implémente un système d'apprentissage fédéré (Federated Learning, FL) pour entraîner le modèle multimodal LLaVA 1.5 (7B) sur des données non organisées provenant de Facebook et Instagram. Il génère des profils spécifiques à chaque plateforme pour chaque client et fusionne ces profils en un profil général. Le projet utilise Kubernetes pour le déploiement, Kubeflow pour l'orchestration, Flower pour l'apprentissage fédéré, CrewAI pour l'orchestration des tâches client, et MCP pour exposer les outils. Les données et le modèle sont stockés via des volumes persistants (PV/PVC).
Architecture du projet
L'architecture est organisée en quatre couches principales : stockage, client, hôte, et orchestration. Voici une description détaillée de chaque couche et de leurs interactions.
a. Couche de stockage
Rôle : Fournit un stockage persistant pour les données (Facebook/Instagram) et le modèle LLaVA.
Composants :

Persistent Volumes (PV) :

data_pv.yaml : Définit un volume de 10 Go (/mnt/data) pour stocker les données brutes (raw_facebook_data.json), intermédiaires (dummy_data.json), et nettoyées (cleaned_data.json).
model_pv.yaml : Définit un volume de 50 Go (/mnt/model) pour stocker le modèle LLaVA (liuhaotian/llava-v1.5-7b).
Backend : Utilise hostPath pour simplifier (à remplacer par NFS, GCS, ou EBS en production).


Persistent Volume Claims (PVC) :

data_pvc.yaml : Permet aux pods clients d'accéder au stockage de données à /data.
model_pvc.yaml : Permet aux pods clients d'accéder au modèle à /model.


Script utilisateur :

upload_data.sh : Copie les données locales (facebook_data.json) vers /data/raw_facebook_data.json dans data-pvc et télécharge le modèle LLaVA vers /model dans model-pvc à l'aide de pods temporaires.



Flux de données :

Les données brutes sont téléversées via upload_data.sh.
Les pods clients lisent/écrivent dans /data via data-pvc.
Le modèle est accédé depuis /model via model-pvc.

b. Couche client
Rôle : Gère les tâches de prétraitement, d'entraînement, et de génération de profils pour chaque client (par exemple, client1 pour Facebook, client2 pour Instagram).
Composants :

Pods clients (deploy_clients.yaml) :

Chaque pod contient deux conteneurs :
mcp-server : Exécute client_mcp_server.py pour fournir des outils via MCP.
flower-client : Exécute flower_client.py pour participer à l'apprentissage fédéré avec Flower.


Montage des volumes :
/data : Lié à data-pvc pour accéder aux données.
/model : Lié à model-pvc pour accéder au modèle LLaVA.
/output : Volume emptyDir pour stocker les poids LoRA et les profils.




Scripts clients :

download_model.py : Télécharge le modèle LLaVA depuis Hugging Face vers /model (exécuté une fois via upload_data.sh).
fetch_data.py : Lit /data/raw_facebook_data.json et extrait les champs pertinents (par exemple, text, image_url) pour produire /data/dummy_data.json.
clean_data.py : Nettoie /data/dummy_data.json pour produire /data/cleaned_data.json avec des champs formatés pour LLaVA (image, text).
train_llava.py : Entraîne le modèle LLaVA avec LoRA sur /data/cleaned_data.json, produisant des poids LoRA spécifiques au client (par exemple, /output/lora_weights_client1).
generate_profile.py : Évalue les poids LoRA sur les données nettoyées pour générer un profil spécifique à la plateforme (par exemple, /output/profile_client1.json).
client_workflow.py : Orchestre les tâches client (téléchargement, récupération, nettoyage, entraînement, génération de profil) via CrewAI.
client_mcp_server.py : Expose les outils (download_model, fetch_data, clean_data, train_llava, generate_profile) via le serveur MCP.
flower_client.py : Implémente le client Flower, gérant l'échange de poids LoRA avec le serveur Flower et exécutant le flux CrewAI.



Flux de travail client :

Télécharge le modèle LLaVA (si nécessaire, bien que pré-téléchargé via upload_data.sh).
Récupère et traite les données brutes (fetch_data.py → clean_data.py).
Entraîne le modèle avec LoRA (train_llava.py).
Génère un profil spécifique à la plateforme (generate_profile.py).
Envoie les poids LoRA au serveur Flower via flower_client.py.

c. Couche hôte
Rôle : Coordonne l'apprentissage fédéré et fusionne les profils spécifiques en un profil général.
Composants :

Pod hôte (deploy_host.yaml) :

Contient un conteneur exécutant mcp_host.py.
Montage du volume /output (emptyDir) pour stocker le modèle global et le profil général.


Scripts hôte :

mcp_host.py :
Exécute le serveur Flower pour coordonner les rounds FL (par exemple, 3 rounds).
Agrège les poids LoRA des clients via la stratégie FedAvg.
Sauvegarde le modèle global agrégé (/output/global).
Fusionne les profils spécifiques des clients (/output/profile_client1.json, /output/profile_client2.json) en un profil général (/output/general_profile.json).


fuse_profiles.py : Effectue la fusion des profils en calculant la moyenne des embeddings.



Flux de travail hôte :

Initialise les paramètres du modèle LLaVA.
Coordonne les rounds FL avec les clients via Flower.
Agrège les poids LoRA pour produire un modèle global.
Fusionne les profils spécifiques en un profil général.

d. Couche d'orchestration
Rôle : Orchestre les tâches des clients et de l'hôte via Kubeflow et déploie le modèle global avec KFServing.
Composants :

Pipeline Kubeflow (fl_pipeline.py) :

Définit deux composants :
flower_client_op : Exécute flower_client.py pour chaque client, produisant des profils (/output/profile_clientX.json).
mcp_host_op : Exécute mcp_host.py, coordonne l'apprentissage fédéré, et fusionne les profils.


Compile en fl_pipeline.yaml pour exécution dans Kubeflow.


KFServing (kfserving.yaml) :

Déploie le modèle global agrégé (/output/global) comme un service d'inférence.



Flux de travail :

Le pipeline Kubeflow lance les tâches client (flower_client_op) pour client1 (Facebook) et client2 (Instagram).
Une fois les tâches client terminées, il lance la tâche hôte (mcp_host_op) pour l'agrégation et la fusion des profils.
Le modèle global est déployé via KFServing pour des inférences futures.

Flux de données global

Téléversement initial :

L'utilisateur exécute upload_data.sh pour copier facebook_data.json dans data-pvc (/data/raw_facebook_data.json) et télécharger le modèle LLaVA dans model-pvc (/model).


Prétraitement client :

Chaque client (client1, client2) lit les données brutes, les nettoie, et produit /data/cleaned_data.json.


Entraînement fédéré :

Les clients entraînent le modèle LLaVA avec LoRA sur /data/cleaned_data.json, produisant des poids LoRA (/output/lora_weights_clientX).
Les poids LoRA sont envoyés au serveur Flower pour agrégation.


Génération de profils :

Chaque client génère un profil spécifique à la plateforme (/output/profile_clientX.json) en évaluant les poids LoRA sur les données nettoyées.


Fusion des profils :

L'hôte fusionne les profils en un profil général (/output/general_profile.json) via fuse_profiles.py.


Déploiement :

Le modèle global est déployé via KFServing pour des inférences futures.



Rôles des fichiers clés

Manifests Kubernetes :

data_pv.yaml, data_pvc.yaml : Stockage des données.
model_pv.yaml, model_pvc.yaml : Stockage du modèle.
deploy_clients.yaml : Déploie les pods clients avec les conteneurs mcp-server et flower-client.
deploy_host.yaml : Déploie le pod hôte avec le serveur Flower.
kfserving.yaml : Déploie le modèle global pour l'inférence.


Scripts clients :

download_model.py : Télécharge le modèle LLaVA depuis Hugging Face.
fetch_data.py : Extrait les champs pertinents des données brutes.
clean_data.py : Nettoie les données pour l'entraînement.
train_llava.py : Entraîne le modèle LLaVA avec LoRA.
generate_profile.py : Génère les profils spécifiques.
client_workflow.py : Orchestre les tâches via CrewAI.
client_mcp_server.py : Expose les outils via MCP.
flower_client.py : Gère l'apprentissage fédéré client.


Scripts hôte :

mcp_host.py : Coordonne l'apprentissage fédéré et la fusion des profils.
fuse_profiles.py : Fusionne les profils.


Pipeline :

fl_pipeline.py : Orchestre les tâches client et hôte.


Script utilisateur :

upload_data.sh : Téléverse les données et le modèle.



Configuration dans VS Code (Linux)
Prérequis

Système : Linux (par exemple, Ubuntu 22.04).
Outils : kubectl, VS Code, Docker, Python 3.12.
Cluster : Accès à un cluster Kubernetes (par exemple, Minikube ou cluster cloud comme GKE) avec un fichier kubeconfig.

Étapes

Configurer kubeconfig :

Copiez le fichier config fourni par le centre de données dans ~/.kube/config :mkdir -p ~/.kube
cp /chemin/vers/config ~/.kube/config
chmod 600 ~/.kube/config


Vérifiez la connexion :kubectl cluster-info
kubectl get nodes




Installer les extensions VS Code :

Kubernetes (ms-kubernetes-tools.vscode-kubernetes-tools) : Gère les ressources Kubernetes.
YAML (redhat.vscode-yaml) : Valide les manifests.
Python (ms-python.python) : Débogue les scripts.
Docker (ms-azuretools.vscode-docker) : Construit les images.


Appliquer les manifests :

Ouvrez le dossier federated_llava_project dans VS Code.
Appliquez les manifests via l'extension Kubernetes ou le terminal :kubectl apply -f manifests/data_pv.yaml
kubectl apply -f manifests/data_pvc.yaml
kubectl apply -f manifests/model_pv.yaml
kubectl apply -f manifests/model_pvc.yaml
kubectl apply -f manifests/deploy_clients.yaml
kubectl apply -f manifests/deploy_host.yaml
kubectl apply -f manifests/kfserving.yaml




Téléverser les données :

Mettez à jour upload_data.sh avec le chemin local de facebook_data.json.
Exécutez :chmod +x scripts/upload_data.sh
./scripts/upload_data.sh




Compiler et exécuter le pipeline :

Configurez l'environnement virtuel :cd federated_llava_project
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements/client_requirements.txt
pip install kfp==2.7.0


Compilez fl_pipeline.py :python pipelines/fl_pipeline.py


Téléversez fl_pipeline.yaml dans l'interface Kubeflow.



Remarques

Stockage : Remplacez hostPath par un backend cloud (NFS, GCS, EBS) pour la production.
Images : Le script train_llava.py peut être modifié pour traiter les images (voir la version mise à jour dans le projet).
Sécurité : Configurez RBAC et chiffrez les données sensibles dans data-pvc.

Pour plus de détails, contactez l'administrateur du projet ou consultez la documentation Kubernetes et Kubeflow.
