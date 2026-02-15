import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import random
import re
import ast


class RecommendationEngine:
    def __init__(self, csv_path, n_clusters=20, n_init=10):
        self.df = pd.read_csv(csv_path)
        
        self.features = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence"]
        self.scaler = StandardScaler()
        self.df_scaled = self.scaler.fit_transform(self.df[self.features])
        

        self.df["clean_title"] = self.df["name"].apply(self._clean_text)
        self.df["main_artist"] = self.df["artists"].apply(lambda x: self._clean_text(eval(x)[0]) if isinstance(x, str) else "")

        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=17)

        self.df["cluster"] = self.kmeans.fit_predict(self.df_scaled)

    def _clean_text(self, text):
        """
        Nettoie le texte pour gérer les feat, remaster, radio edit, etc.
        Ex: "Poker Face (feat. Kid Cudi) - Radio Edit" -> "poker face"
        """
        if not isinstance(text, str): 
            return ""

        text = text.lower()
        text = re.sub(r'\s*[\(\[].*?[\)\]]', '', text)
        text = re.sub(r'\s-.*', '', text)

        return text.strip()

    def get_audio_vector(self, idx):
        """Récupère le vecteur audio normalisé d'une chanson par son index"""
        return self.df_scaled[idx].reshape(1, -1)

    def is_duplicate(self, idx_a, idx_b, threshold=0.90):
        """
        Vérifie si deux chansons sont des doublons en utilisant :
        1. Le titre simplifié (Doit être identique)
        2. La similarité audio (Doit être très haute > 0.90)
        """

        #On vérifie les titres
        title_a = self.df.iloc[idx_a]["clean_title"]
        title_b = self.df.iloc[idx_b]["clean_title"]
        
        if title_a != title_b:
            return False

        #On vérifie l'artiste principal    
        artist_a = self.df.iloc[idx_a]["main_artist"]
        artist_b = self.df.iloc[idx_b]["main_artist"]
        
        if artist_a != artist_b:
            return False

        # On check las similarité des deux musiques
        vec_a = self.get_audio_vector(idx_a)
        vec_b = self.get_audio_vector(idx_b)
        sim = cosine_similarity(vec_a, vec_b)[0][0]
        
        return sim > threshold

    def post_process_playlist(self, candidate_indices):
        """
        Nettoie une liste d'indices candidats pour retirer les doublon
        Garde la première occurrence rencontrée
        """
        final_indices = []
        seen_indices = []
        
        for idx in candidate_indices:

            is_new = True
            for accepted_idx in seen_indices:

                if self.is_duplicate(idx, accepted_idx):

                    is_new = False
                    break
            
            if is_new:

                final_indices.append(idx)
                seen_indices.append(idx)
                
        return self.df.iloc[final_indices][["name", "artists", "year", "cluster", "popularity"]].copy()


    # def recommend_artist_affinity(self, user, n=5):
    #     """
    #     Stratégie : Exploitation pure
    #     Recommande des titres non écoutés des artistes favoris de l'utilisateur
    #     """
    #     #On récupère les artistes préférés de l'utilisateur
    #     top_artists = user.get_top_artists(n=10) 
    #     if not top_artists: 
    #         return []

    #     recommendations = []
    #     random.shuffle(top_artists)
        
    #     for artist in top_artists:

    #         mask = (self.df["artists"].str.contains(artist, case=False, regex=False))
    #         candidates = self.df[mask]
            
    #         for idx in candidates.index:
    #             if len(recommendations) >= n: 
    #                 break
                
    #             if idx not in user.liked_indices:

    #                 recommendations.append(idx)
        
    #     return recommendations[:n]
    

    def recommend_artist_affinity(self, user, n=5):
        """
        Stratégie : Exploitation pure
        Recommande des titres non écoutés des artistes favoris de l'utilisateur
        """

        top_artists = user.get_top_artists(n=10)
        if not top_artists: return []

        recommendations = []
        random.shuffle(top_artists)
        

        candidates_per_artist = {}
        for artist in top_artists:
            mask = (self.df["artists"].str.contains(artist, case=False, regex=False))

            available = self.df[mask & ~self.df.index.isin(user.liked_indices)]

            if not available.empty:
                candidates_per_artist[artist] = available.sort_values('popularity', ascending=False).index.tolist()

        while len(recommendations) < n and candidates_per_artist:
            artists_to_remove = []
            
            for artist in list(candidates_per_artist.keys()):
                if len(recommendations) >= n: break
                
                song_idx = candidates_per_artist[artist].pop(0)
                recommendations.append(song_idx)
                
                if not candidates_per_artist[artist]:
                    artists_to_remove.append(artist)
            
            for artist in artists_to_remove:
                del candidates_per_artist[artist]
                
        return recommendations[:n]


    def recommend_collaborative(self, user, user_pool, n=5, neighbor_threshold=0.1):
        """
        Stratégie : Filtrage collaboratif User-User.
        Score(Chanson) = Somme(Similarité(Voisin) * a_aimé)
        """
        scores = {}
        
        for neighbor in user_pool:
            #On s'exclu de la comparaison
            if neighbor.user_id == user.user_id: 
                continue 
            
            sim = user.get_similarity(neighbor, self)
            
            if sim < neighbor_threshold: 
                continue 
            
            for idx in neighbor.liked_indices:
                if idx not in user.liked_indices:
                    scores[idx] = scores.get(idx, 0) + sim
                    
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [idx for idx, score in sorted_indices[:n]]
    

    def recommend_content_based(self, user, n=5):
        """
        Stratégie : Similarité Cosinus sur les features audio
        """
        if not user.liked_indices: return []

        indices = list(user.liked_indices)
        user_vectors = self.df_scaled[indices]
        user_centroid = np.mean(user_vectors, axis=0).reshape(1, -1)
        
        sim_scores = cosine_similarity(user_centroid, self.df_scaled)[0]
        
        sorted_indices = sim_scores.argsort()[::-1]
        
        recommendations = []
        for idx in sorted_indices:
            if len(recommendations) >= n: break
            if idx not in user.liked_indices:
                recommendations.append(idx)
                
        return recommendations


    def recommend_exploration(self, user, n=5):
        """
        Stratégie : Exploration
        Si pas d'historique : Propose un échantillon représentatif (Top popularité par cluster)
        Sinon : Identifie les clusters audio ignorés et propose des sons populaire de ces clusters
        """
        #Nouvel utilisateur
        if not user.liked_indices:
            candidates = []
        
            available_clusters = list(range(self.n_clusters))
            random.shuffle(available_clusters)
            
            for cluster_id in available_clusters:

                if len(candidates) >= n:
                    break
                
                top_cluster = self.df[self.df["cluster"] == cluster_id].nlargest(10, "popularity")
                
                if not top_cluster.empty:
                    candidates.append(top_cluster.sample(1).index[0])
            
            # Si jamais on n'a pas assez de clusters on complète avec du random populaire
            if len(candidates) < n:

                remaining = n - len(candidates)
                extras = self.df.nlargest(100, "popularity").sample(remaining).index.tolist()

                candidates.extend(extras)
                
            return candidates

        # User avec un historique
        liked_clusters = self.df.loc[list(user.liked_indices), "cluster"].unique()
        
        all_clusters = set(range(self.n_clusters))
        ignored_clusters = list(all_clusters - set(liked_clusters))
        
        if not ignored_clusters: 
            ignored_clusters = list(all_clusters)
        
        target_cluster = random.choice(ignored_clusters)
        
        candidates = self.df[(self.df["cluster"] == target_cluster) & (self.df["popularity"] > 50)]
        
        candidates = candidates[~candidates.index.isin(user.liked_indices)]
        
        if candidates.empty: 
            return []
        
        return candidates.sample(min(n, len(candidates))).index.tolist()


    def generate_hybrid_playlist(self, user, user_pool, total_songs=10, weights=(0.2, 0.3, 0.4, 0.1)):
        """
        Génère la playlist finale
        Gère intelligemment les nouveaux utilisateurs.
        """
  
        # Si l'utilisateur n'a rien liké on ignore les poids demandés et on met tout sur l'exploration Z
        if not user.liked_indices:
            return self.df.loc[self.recommend_exploration(user, n=total_songs)][["name", "artists", "year", "popularity"]]

        # Utilisateur avec historique
        w, x, y, z = weights
        n_w = int(total_songs * w)
        n_x = int(total_songs * x)
        n_y = int(total_songs * y)
        n_z = total_songs - (n_w + n_x + n_y)
        
        recos_w = self.recommend_artist_affinity(user, n=n_w + 2)
        recos_x = self.recommend_collaborative(user, user_pool, n=n_x + 2)
        recos_y = self.recommend_content_based(user, n=n_y + 2)
        recos_z = self.recommend_exploration(user, n=n_z + 2)
        
        raw_indices = recos_x[:n_x] + recos_z[:n_z] + recos_w[:n_w] + recos_y[:n_y]
        
        current_len = len(raw_indices)
        if current_len < total_songs:
            missing = total_songs - current_len
            makeup = self.recommend_content_based(user, n=missing + 5)
            if not makeup:
                makeup = self.recommend_exploration(user, n=missing + 5)
                
            raw_indices.extend(makeup)
            
        final_df = self.post_process_playlist(raw_indices)
        final_selection = final_df.head(total_songs)
        
        return final_selection.sample(frac=1).reset_index(drop=True)


    def get_genre_anchors(self):
        """
        Crée des signatures sonores de référence) pour chaque genre
        en se basant sur des artistes incontestables du dataset
        """
        anchors_definitions = {
            'Rock': ['Queen', 'Led Zeppelin', 'AC/DC', 'Nirvana', 'The Rolling Stones', 'Metallica'],
            'Pop': ['Taylor Swift', 'Ariana Grande', 'Katy Perry', 'Ed Sheeran', 'Bruno Mars', "Rihanna"],
            'Rap/Hip-Hop': ['Eminem', '2Pac', 'Drake', 'Kendrick Lamar', 'Juice WRLD', 'Jay-Z', 'Snoop Dogg', "XXXTENTACION"],
            'Jazz/Soul': ['Miles Davis', 'Nina Simone', 'Frank Sinatra', 'Louis Armstrong'],
            'Electro': ['Daft Punk', 'Avicii', 'deadmau5', 'Marshmello', 'Skrillex'],
            'Classique': ['Bach', 'Beethoven', 'Mozart', 'Choppin', 'Vivaldi']
        }
        
        genre_signatures = {}
        
        for genre, artists in anchors_definitions.items():

            pattern = '|'.join(artists)
            mask = self.df["artists"].str.contains(pattern, case=False, na=False)
            
            if mask.any():
                vectors = self.df_scaled[mask]
                genre_signatures[genre] = np.mean(vectors, axis=0)
        
        return genre_signatures

class User:

    def __init__(self, user_id):
        self.user_id = user_id
        
        self.liked_indices = set()
        
        self.liked_keys = set()
        
        self.favorite_artists = {}

    def add_liked_song(self, row, index):
        """
        Ajoute une chanson à l'historique de l'utilisateur
        :param row: La ligne du DataFrame
        :param index: L'index de la chanson dans le DataFrame
        """
        if "unique_key" in row:

            key = row["unique_key"]
        else:
            key = str(row["name"]).lower()
            
        if key in self.liked_keys:
            return
            
        self.liked_keys.add(key)
        self.liked_indices.add(index)
        
        try:
            if isinstance(row["artists"], str):
                artist_list = ast.literal_eval(row["artists"])
            else:
                artist_list = row["artists"]
                
            for artist in artist_list:
                artist_clean = artist.lower().strip()
                self.favorite_artists[artist_clean] = self.favorite_artists.get(artist_clean, 0) + 1
        except:
            pass


    def remove_liked_song(self, row, index):
        """
        Supprime une chanson de l'historique et met à jour les préférences
        """
        key = row["unique_key"] if "unique_key" in row else str(row["name"]).lower()
        
        
        if key not in self.liked_keys:
            return 
            

        self.liked_keys.remove(key)
        self.liked_indices.remove(index)
        
        try:
            if isinstance(row["artists"], str):
                artist_list = ast.literal_eval(row["artists"])
            else:
                artist_list = row["artists"]
                
            for artist in artist_list:
                artist_clean = artist.lower().strip()
                if artist_clean in self.favorite_artists:
                    self.favorite_artists[artist_clean] -= 1
                    
                    if self.favorite_artists[artist_clean] <= 0:
                        del self.favorite_artists[artist_clean]
        except:
            pass


    def get_similarity(self, other_user, engine, weights=(0.2, 0.5, 0.3)):
        """
        Calcule une similarité hybride entre deux utilisateurs.
        weights = (poids_chansons, poids_artistes, poids_audio)
        """
        w_song, w_artist, w_audio = weights
        
        # similarité des chansons 
        inter_songs = len(self.liked_keys.intersection(other_user.liked_keys))
        union_songs = len(self.liked_keys.union(other_user.liked_keys))
        song_sim = inter_songs / union_songs if union_songs > 0 else 0
        
        # similarité des artistes 
        artists_self = set(self.favorite_artists.keys())
        artists_other = set(other_user.favorite_artists.keys())
        inter_art = len(artists_self.intersection(artists_other))
        union_art = len(artists_self.union(artists_other))
        artist_sim = inter_art / union_art if union_art > 0 else 0
        
        # similarité du rofil audio
        audio_sim = 0
        if self.liked_indices and other_user.liked_indices:

            vec_self = np.mean(engine.df_scaled[list(self.liked_indices)], axis=0).reshape(1, -1)
            vec_other = np.mean(engine.df_scaled[list(other_user.liked_indices)], axis=0).reshape(1, -1)

            audio_sim = cosine_similarity(vec_self, vec_other)[0][0]
            audio_sim = max(0, audio_sim)
            
        total_similarity = (w_song * song_sim) + (w_artist * artist_sim) + (w_audio * audio_sim)
        return total_similarity

    def get_top_artists(self, n=5):
        """Retourne la liste des n artistes préférés"""
        sorted_artists = sorted(self.favorite_artists.items(), key=lambda x: x[1], reverse=True)

        return [artist for artist, count in sorted_artists[:n]]


    def get_musical_identity(self, engine):
        """
        Analyse les likes de l'utilisateur pour définir son identité
        Retourne le genre dominant ou un mélange de deux genres
        """
        if not self.liked_indices:
            return "Nouvel utilisateur"

        user_vector = np.mean(engine.df_scaled[list(self.liked_indices)], axis=0).reshape(1, -1)
        
        anchors = engine.get_genre_anchors()
        
        scores = []
        for genre, anchor_vec in anchors.items():
            sim = cosine_similarity(user_vector, anchor_vec.reshape(1, -1))[0][0]
            scores.append((genre, sim))
        
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        top_1, score_1 = scores[0]
        top_2, score_2 = scores[1]
        
        if (score_1 - score_2) < 0.12:
            return f"{top_1} & {top_2}"
        else:
            return top_1

    def plot_profile(self, engine):
        """
        Affiche un graphique radar représentant la moyenne des caractéristiques 
        audio des chansons aimées par l'utilisateur
        """
        if not self.liked_indices:
            print(f"L'utilisateur {self.user_id} n'a pas encore de profil audio (0 likes)")
            return

        features_list = engine.features

        vectors = engine.df_scaled[list(self.liked_indices)]
        mean_vector = np.mean(vectors, axis=0)
        

        
        angles = np.linspace(0, 2 * np.pi, len(features_list), endpoint=False).tolist()

        values = mean_vector.tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="green", alpha=0.25)
        ax.plot(angles, values, color="green", linewidth=2)

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features_list)
        
        plt.title(f"Profil Musical : {self.user_id}", size=15, color="green", y=1.1)
        plt.show()



def find_and_add(nom_chanson, nom_artiste, user, engine):
    mask = (engine.df["name"].str.contains(nom_chanson, case=False, na=False)) & \
           (engine.df["artists"].str.contains(nom_artiste, case=False, na=False))
    
    resultats = engine.df[mask]
    
    if not resultats.empty:
        idx = resultats.index[0]
        row = resultats.iloc[0]
        user.add_liked_song(row, idx)
        print(f"Ajouté : {row['name']} - {row['artists']}")
    else:
        print(f"Non trouvé : {nom_chanson} par {nom_artiste}")

def compare_user(u1, u2):
    sim = u1.get_similarity(u2, engine)
    communs = u1.liked_keys.intersection(u2.liked_keys)
    
    print(f"--- Comparaison : {u1.user_id} vs {u2.user_id} ---")
    print(f"Score de similarité : {sim:.2%}")
    print(f"Nombre de titres en commun : {len(communs)}")
    if communs:
        print(f"Exemples : {list(communs)[:3]}")
    print("-" * 40)



def plot_profiles_side_by_side(users, engine):
    n = len(users)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6), subplot_kw=dict(polar=True))
    if n == 1: axes = [axes]
    
    features = engine.features
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    for i, user in enumerate(users):
        vectors = engine.df_scaled[list(user.liked_indices)]
        mean_v = np.mean(vectors, axis=0).tolist()
        mean_v += mean_v[:1]
        
        axes[i].fill(angles, mean_v, color="green", alpha=0.3)
        axes[i].plot(angles, mean_v, color="green", linewidth=2)
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels(features, fontsize=9)
        axes[i].set_title(f"Profil : {user.user_id}", size=14, pad=20)

    plt.tight_layout()
    plt.show()