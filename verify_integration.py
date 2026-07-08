import torch
from net.st_gcn import Model
from net.body_part import BodyPartAggregator
from net.kg_gnn import KG_GNN
from net.interaction import DynamicBodyInteraction, TemporalEnergyAttention

def verify():
    print("Verifying imports...")
    print("net.interaction imported successfully.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Init model
    print("Building model...")
    graph_args = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    model = Model(
        in_channels=3,
        num_class=120,
        graph_args=graph_args,
        edge_importance_weighting=True,
        use_kg=True,
        dropout=0.5,
    ).to(device)
    model.eval()
    print("Model built successfully.")

    # We want to print shapes. We can use forward hooks.
    shapes = {}
    
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                shapes[name] = output[0].shape
            else:
                shapes[name] = output.shape
        return hook

    model.body_part.register_forward_hook(get_hook('BodyPartAggregator'))
    model.kg_gnn.register_forward_hook(get_hook('KG_GNN'))
    model.dynamic_interaction.register_forward_hook(get_hook('DynamicBodyInteraction'))
    model.temporal_attention.register_forward_hook(get_hook('TemporalEnergyAttention'))
    
    # Create dummy tensor: (N, C, T, V, M)
    N, C, T, V, M = 2, 3, 300, 25, 2
    x = torch.randn(N, C, T, V, M).to(device)
    print(f"\n--- Shape Verification ---")
    print(f"Input: {x.shape}")
    
    # Wrap backbone
    orig_backbone = model._forward_backbone
    def mock_backbone(*args, **kwargs):
        out = orig_backbone(*args, **kwargs)
        shapes['Backbone'] = out[0].shape
        return out
    model._forward_backbone = mock_backbone

    try:
        with torch.no_grad():
            out = model(x)
        shapes['Final classifier'] = out.shape
        
        print(f"Backbone: {shapes.get('Backbone')}")
        print(f"DynamicBodyInteraction: {shapes.get('DynamicBodyInteraction')}")
        print(f"TemporalEnergyAttention: {shapes.get('TemporalEnergyAttention')}")
        print(f"BodyPartAggregator: {shapes.get('BodyPartAggregator')}")
        print(f"KG_GNN: {shapes.get('KG_GNN')}")
        print(f"Final classifier: {shapes.get('Final classifier')}")
        print("\n[SUCCESS] Forward pass completed without errors or mismatches!")
    except Exception as e:
        print(f"\n[ERROR] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verify()
