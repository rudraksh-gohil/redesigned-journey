# Script 1: Data Extraction and Preprocessing

import torch
from neo4j import GraphDatabase
from torch_geometric.data import Data

# Neo4j credentials
URI = "neo4j+ssc://75ecede3.databases.neo4j.io"
AUTH = ("neo4j", "2j7n35A11vZB_wloMpOBB0GDn0qHaXQYYbxhNU6hxzI")

# Connect to Neo4j database
def connect_to_neo4j():
    """Connect to the Neo4j database."""
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        driver.verify_connectivity()
        print("Successfully connected to Neo4j!")
        return driver
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return None

driver = connect_to_neo4j()

# Extract graph data from Neo4j
def extract_graph_data(driver):
    """Extract nodes and relationships from Neo4j."""
    query_nodes = """
    MATCH (n)
    RETURN id(n) AS node_id, labels(n) AS labels, properties(n) AS properties
    """
    query_relationships = """
    MATCH (n)-[r]->(m)
    RETURN id(n) AS source, id(m) AS target, type(r) AS relationship_type
    """

    with driver.session() as session:
        nodes_result = session.run(query_nodes)
        nodes = [{"id": record["node_id"], "labels": record["labels"], "properties": record["properties"]} for record in nodes_result]

        relationships_result = session.run(query_relationships)
        relationships = [{"source": record["source"], "target": record["target"], "type": record["relationship_type"]} for record in relationships_result]

    return nodes, relationships

# Generate node features based on labels and properties
def generate_node_features(nodes):
    """Generate feature vectors based on node type-specific properties."""
    feature_list = []

    for node in nodes:
        node_labels = node["labels"]
        node_props = node["properties"]

        if "RequirementType" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("type", "")) % 1000], dtype=torch.float))

        elif "AcceptanceCriteria" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("acceptance_criteria", "")) % 1000], dtype=torch.float))

        elif "AppName" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("app_name", "")) % 1000], dtype=torch.float))

        elif "CommonBug" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("commonbugs", "")) % 1000], dtype=torch.float))

        elif "Domain" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("domain", "")) % 1000], dtype=torch.float))

        elif "Feature" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("feature_name", "")) % 1000], dtype=torch.float))

        elif "Platform" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("platform_name", "")) % 1000], dtype=torch.float))

        elif "Quality" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("quality", "")) % 1000], dtype=torch.float))

        elif "Region" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("region_name", "")) % 1000], dtype=torch.float))

        elif "SoftwareType" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("software_type", "")) % 1000], dtype=torch.float))

        elif "Subdomain" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("subdomain_name", "")) % 1000], dtype=torch.float))

        elif "UserStory" in node_labels:
            feature_list.append(torch.tensor([hash(node_props.get("user_story", "")) % 1000], dtype=torch.float))

        else:
            feature_list.append(torch.tensor([0.0], dtype=torch.float))

    return torch.stack(feature_list)

# Generate edge indices and attributes
def generate_edges_and_attributes(relationships, node_id_map):
    edge_index = [(node_id_map[rel["source"]], node_id_map[rel["target"]]) for rel in relationships]
    edge_attr = [hash(rel["type"]) % 1000 for rel in relationships]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr

# Convert to PyTorch Geometric format
def convert_to_pyg(nodes, relationships):
    node_id_map = {node["id"]: idx for idx, node in enumerate(nodes)}

    # Generate node features
    x = generate_node_features(nodes)

    # Generate edge indices and attributes
    edge_index, edge_attr = generate_edges_and_attributes(relationships, node_id_map)

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# Main logic
if driver:
    nodes, relationships = extract_graph_data(driver)
    print(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships from Neo4j.")

    # Convert to PyTorch Geometric format
    pyg_data = convert_to_pyg(nodes, relationships)
    print("Converted data to PyG format:")
    print(f"Node features shape: {pyg_data.x.shape}")
    print(f"Edge index shape: {pyg_data.edge_index.shape}")
    print(f"Edge attributes shape: {pyg_data.edge_attr.shape}")

    # Save PyG data to file
    torch.save(pyg_data, "graph_data.pt")
    print("Graph data saved to 'graph_data.pt'.")
else:
    print("Failed to connect to Neo4j.")
