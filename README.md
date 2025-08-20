# 📌 Projet d'Apprentissage Fédéré avec LLaVA

Ce projet implémente un système d'apprentissage fédéré (Federated Learning, FL) pour entraîner le modèle multimodal **LLaVA 1.5 (7B)** sur des données non organisées provenant de **Facebook** et **Instagram**.  
Il génère des profils spécifiques à chaque plateforme pour chaque client et fusionne ces profils en un profil général.  

Le projet utilise :  
- **Kubernetes** pour le déploiement  
- **Kubeflow** pour l'orchestration  
- **Flower** pour l'apprentissage fédéré  
- **CrewAI** pour l'orchestration des tâches client  
- **MCP** pour exposer les outils  
- **KFServing** pour le déploiement du modèle global  

Les données et le modèle sont stockés via des volumes persistants (PV/PVC).

---

## 🏗️ Architecture du projet

L'architecture est organisée en **quatre couches principales** : **Stockage, Client, Hôte, Orchestration**.  

### 1. Couche de stockage

**Rôle :** Fournit un stockage persistant pour les données (Facebook/Instagram) et le modèle LLaVA.

#### Composants :
- **Persistent Volumes (PV)**  
  - `data_pv.yaml` : 10 Go (`/mnt/data`) pour les données brutes, intermédiaires, nettoyées.  
  - `model_pv.yaml` : 50 Go (`/mnt/model`) pour stocker le modèle **liuhaotian/llava-v1.5-7b**.  
  - Backend : `hostPath` (à remplacer par NFS, GCS ou EBS en production).

- **Persistent Volume Claims (PVC)**  
  - `data_pvc.yaml` : accès aux données via `/data`.  
  - `model_pvc.yaml` : accès au modèle via `/model`.

- **Script utilisateur**  
  - `upload_data.sh` : copie `facebook_data.json` vers `/data/raw_facebook_data.json` et télécharge le modèle vers `/model`.

#### Flux de données :
1. L’utilisateur téléverse les données avec `upload_data.sh`.  
2. Les pods clients lisent/écrivent dans `/data` via `data-pvc`.  
3. Le modèle est accédé dans `/model` via `model-pvc`.

---

### 2. Couche client

**Rôle :** Prétraitement, entraînement, génération de profils spécifiques à la plateforme (Facebook, Instagram).  

#### Composants :
- **Pods clients (`deploy_clients.yaml`)** :  
  - Conteneur `mcp-server` → `client_mcp_server.py`  
  - Conteneur `flower-client` → `flower_client.py`  
- **Volumes montés** :  
  - `/data` → `data-pvc`  
  - `/model` → `model-pvc`  
  - `/output` → `emptyDir` (poids LoRA + profils)

#### Scripts :
- `download_model.py` → Télécharge le modèle Hugging Face.  
- `fetch_data.py` → Extrait les champs pertinents (texte, image_url).  
- `clean_data.py` → Nettoie les données en JSON structuré.  
- `train_llava.py` → Entraîne LLaVA avec LoRA, génère `lora_weights_clientX`.  
- `generate_profile.py` → Produit `profile_clientX.json`.  
- `client_workflow.py` → Orchestre via CrewAI.  
- `client_mcp_server.py` → Expose les outils (MCP).  
- `flower_client.py` → Participe à l’apprentissage fédéré.  

#### Flux client :

---

### 3. Couche hôte

**Rôle :** Coordonne l'apprentissage fédéré et fusionne les profils en un profil général.  

#### Composants :
- **Pod hôte (`deploy_host.yaml`)**  
  - Conteneur : `mcp_host.py`  
  - Volume : `/output` (modèle global + profil général)

#### Scripts :
- `mcp_host.py` → serveur Flower, agrégation FedAvg, fusion des profils.  
- `fuse_profiles.py` → moyenne des embeddings pour générer `general_profile.json`.

#### Flux hôte :
1. Lance le serveur Flower.  
2. Agrège les poids LoRA (`FedAvg`).  
3. Sauvegarde le modèle global (`/output/global`).  
4. Fusionne les profils → `/output/general_profile.json`.

---

### 4. Couche d'orchestration

**Rôle :** Orchestration avec Kubeflow, déploiement du modèle global avec KFServing.  

#### Composants :
- **Pipeline Kubeflow (`fl_pipeline.py`)**  
  - `flower_client_op` : lance Flower client.  
  - `mcp_host_op` : lance Flower host + fusion.  
  - Compile en `fl_pipeline.yaml`.  

- **KFServing (`kfserving.yaml`)**  
  - Déploie le modèle global pour inférence.

---

## 🔄 Flux global des données

1. **Téléversement initial** → `upload_data.sh` charge `facebook_data.json` et le modèle Hugging Face.  
2. **Prétraitement client** → données brutes → données nettoyées.  
3. **Entraînement FL** → LoRA clients → agrégation (FedAvg).  
4. **Génération de profils** → `profile_clientX.json`.  
5. **Fusion hôte** → `general_profile.json`.  
6. **Déploiement** → KFServing publie le modèle global.

---

## 📂 Rôles des fichiers

### Manifests Kubernetes
- `data_pv.yaml`, `data_pvc.yaml` → stockage données  
- `model_pv.yaml`, `model_pvc.yaml` → stockage modèle  
- `deploy_clients.yaml` → pods clients  
- `deploy_host.yaml` → pod hôte  
- `kfserving.yaml` → déploiement modèle global  

### Scripts clients
- `download_model.py`, `fetch_data.py`, `clean_data.py`  
- `train_llava.py`, `generate_profile.py`  
- `client_workflow.py`, `client_mcp_server.py`, `flower_client.py`  

### Scripts hôte
- `mcp_host.py`, `fuse_profiles.py`  

### Pipeline
- `fl_pipeline.py`  

### Utilitaire
- `upload_data.sh`  

---

## ⚙️ Configuration dans VS Code (Linux)

### Prérequis
- **OS** : Ubuntu 22.04  
- **Outils** : `kubectl`, VS Code, Docker, Python 3.12  
- **Cluster** : Kubernetes (Minikube ou Cloud) + kubeconfig  

### Étapes
1. **Configurer kubeconfig**  
   ```bash
   mkdir -p ~/.kube
   cp /chemin/vers/config ~/.kube/config
   chmod 600 ~/.kube/config
   kubectl cluster-info
   kubectl get nodes
