from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
import networkx as nx
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from langchain_core.documents import Document

# Função para plotar o grafo
def plot_graph(nodes, relationships):
    G = nx.DiGraph()
    
    # Adicionar nós ao grafo
    for node in nodes:
        G.add_node(node.id, label=node.id)
    
    # Adicionar relações ao grafo
    for relationship in relationships:
        G.add_edge(relationship.source.id, relationship.target.id, label=relationship.type)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    # Desenhar nós
    nx.draw_networkx_nodes(G, pos, node_size=7000, node_color='lightblue')
    
    # Desenhar arestas
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrows=True)
    
    # Desenhar rótulos dos nós
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)
    
    # Desenhar rótulos das arestas
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Grafo de Entidades e Relações")
    plt.show()
    
    num_nodes = G.number_of_nodes() 
    num_edges = G.number_of_edges()
    print ( f'Número de nós: {num_nodes} ' ) 
    print ( f'Número de arestas: {num_edges} ' ) 
    print ( f'Proporção entre arestas e nós: { round (num_edges / num_nodes, 2 )} ' )
    
    degree_centrality = nx.degree_centrality(G)
    for node,central in degree_centrality.items():
        print(f' {node} : Grau de centralidade = {central: .2f} ' )
        
    closeness_centrality = nx.closeness_centrality(G)
    for node, central in closeness_centrality.items(): 
        print(f'Centralidade de proximidade de {node} : {central: .2f} ' )
    
    # Calcular medidas de centralidade
    degree_centrality = nx.degree_centrality(G) 
    betweenness_centrality = nx.betweenness_centrality(G) 
    closeness_centrality = nx.closeness_centrality(G) 

    # Visualizar medidas de centralidade
    plt.figure(figsize=( 15 , 10 )) 

    # Centralidade de grau
    plt.subplot( 131 ) 
    nx.draw(G, pos , with_labels=True, font_size= 10 , node_size=[v * 3000  for v in degree_centrality.values()], node_color=list(degree_centrality.values()), cmap=plt.cm.Blues, edge_color= 'gray' , alpha= 0.6 ) 
    plt.title( 'Grau da Centralidade' ) 

    # Centralidade de intermediação
    plt.subplot( 132 ) 
    nx.draw(G, pos , with_labels=True, font_size= 10 , node_size=[v * 3000  for v in betweenness_centrality.values()], node_color=list(betweenness_centrality.values()), cmap=plt.cm.Oranges, edge_color= 'gray' , alpha= 0.6 ) 
    plt.title( 'Centralidade de intermediação') 

    # Centralidade de proximidade
    plt.subplot( 133 ) 
    nx.draw(G, pos , with_labels=True, font_size= 10 , node_size=[v * 3000  for v in closeness_centrality.values()], node_color=list(closeness_centrality.values()), cmap=plt.cm.Greens, edge_color= 'gray' , alpha= 0.6 ) 
    plt.title( 'Centralidade de proximidade') 

    plt.tight_layout() 
    plt.show()
    
    source_node = 'Edson Arantes do Nascimento'
    target_node = 'Mundiais Interclubes'

    # Find the shortest path
    shortest_path = nx.shortest_path(G, source=source_node, target=target_node)

    # Visualize the shortest path
    plt.figure(figsize=(10, 8))
    path_edges = [(shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)]
    nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color='lightblue', edge_color='gray', alpha=0.6)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    plt.title(f'Shortest Path from {source_node} to {target_node}')
    plt.show()
    print('Shortest Path:', shortest_path)

    from node2vec import Node2Vec

    # Generate node embeddings using node2vec
    nodevec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4) # You can adjust these parameters
    model = nodevec.fit(window=10, min_count=1, batch_words=4) # Training the model

    # Visualize node embeddings using t-SNE
    from sklearn.manifold import TSNE
    import numpy as np

    # Get embeddings for all nodes
    embeddings = np.array([model.wv[node] for node in G.nodes()])

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, perplexity=10, n_iter=400)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Visualize embeddings in 2D space with node labels
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', alpha=0.7)

    # Add node labels
    for i, node in enumerate(G.nodes()):
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], node, fontsize=8)
    plt.title('Node Embeddings Visualization')
    plt.show()

    from sklearn.cluster import KMeans
    
    # Perform K-Means clustering on node embeddings
    num_clusters = 3 # Adjust the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Visualize K-Means clustering in the embedding space with node labels
    plt.figure(figsize=(12, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap=plt.cm.Set1, alpha=0.7)

    # Add node labels
    for i, node in enumerate(G.nodes()):
        plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], node, fontsize=8)

    plt.title('K-Means Clustering in Embedding Space with Node Labels')

    plt.colorbar(label="Cluster Label")
    plt.show()

    from sklearn.cluster import KMeans

    # Perform K-Means clustering on node embeddings
    num_clusters = 3 # Adjust the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Visualize clusters
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color=cluster_labels, cmap=plt.cm.Set1, edge_color='gray', alpha=0.6)
    plt.title('Graph Clustering using K-Means')

    plt.show()

    from sklearn.cluster import DBSCAN

    # Perform DBSCAN clustering on node embeddings
    dbscan = DBSCAN(eps=1.0, min_samples=2) # Adjust eps and min_samples
    cluster_labels = dbscan.fit_predict(embeddings)

    # Visualize clusters
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, font_size=10, node_size=700, node_color=cluster_labels, cmap=plt.cm.Set1, edge_color='gray', alpha=0.6)
    plt.title('Graph Clustering using DBSCAN')
    plt.show() 


# Carregar as variáveis de ambiente
load_dotenv()

# Configuração do modelo genAI
GEMINI_MODEL_NAME = "gemini-pro"

# Configuração do modelo genAI usando LangChain
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

# Definir o modelo e o prompt
llm = GoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=api_key, temperature=0.6)

llm_transformer = LLMGraphTransformer(llm=llm)

text = """Edson Arantes do Nascimento, mais conhecido como Pelé, nasceu em Três Corações na cidade de São Paulo em 23 de outubro de 1940. 
Pelé foi um futebolista brasileiro que atuou como atacante, 
Pelé foi descrito como o Rei do Futebol e é amplamente considerado como o maior atleta de todos os tempos. 
Em 2000, Pelé foi eleito Jogador do Século pela Federação Internacional de História e Estatísticas do Futebol (IFFHS). 
Nesse mesmo ano, Pelé foi eleito Atleta do Século pelo Comitê Olímpico Internacional. 
De acordo com a IFFHS, Pelé é o quinto maior goleador da história do futebol em jogos oficiais, tendo marcado 767 gols em 812 partidas. 
No total foram 1283 gols em 1363 jogos, incluindo amistosos não oficiais, um recorde mundial do Guinness. 
Pelé começou a jogar pelo Santos Futebol Clube aos quinze anos de idade, e pela Seleção Brasileira aos dezesseis, 
Pelé sagrando-se campeão de três edições da Copa do Mundo FIFA: 1958, 1962 e 1970, sendo o único a fazê-lo como jogador. 
Contando os gols oficiais, Pelé é o segundo maior goleador da história da Seleção Brasileira, com 77 gols em 92 jogos. 
Em clubes, Pelé é o maior artilheiro da história do Santos e o levou a várias conquistas, com destaque para duas Copas Libertadores da América e dois Mundiais Interclubes, vencidos em 1962 e 1963. 
Após se aposentar em 1977, Pelé tornou-se embaixador mundial do futebol e fez muitos trabalhos de atuação e comerciais. 
Em janeiro de 1995, Pelé foi nomeado ministro do esporte no governo Fernando Henrique Cardoso. 
Em seus últimos anos de vida, Pelé sofria de câncer de cólon, diagnosticado em 2021. 
Pelé morreu, no dia 29 de dezembro de 2022 na capital paulista.
"""
# Processar documentos e criar grafos
documents = [Document(page_content=text)]

graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Plotar os gráficos para os grafos gerados
print("Grafo original:")

print(f"Nodes:{graph_documents[0].nodes}")
print(f"Relationships:{graph_documents[0].relationships}")

     
# Plotar os gráficos para os grafos gerados
print("Grafo original:")
plot_graph(graph_documents[0].nodes, graph_documents[0].relationships)