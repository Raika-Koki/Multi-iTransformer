from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='Training and Evaluation Workflow')

# Training Loop
dot.node('A', 'Start Training')
dot.node('B', 'Split Train Data into Mini-Batches')
dot.node('C', 'Forward Pass')
dot.node('D', 'Calculate Loss (MSE)')
dot.node('E', 'Backward Pass (Gradient Calculation)')
dot.node('F', 'Update Parameters')
dot.node('G', 'End of Epoch/Trial')

# Add edges for training loop
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

# Validation
dot.node('G1', 'Use Valid Data to Calculate Loss (MSE)')
dot.node('G2', 'Check if Validation Loss is Minimum')
dot.node('G3', 'Save Optimal Model Parameters')
dot.node('G4', 'Continue Training or Stop')

# Add edges for validation
dot.edge('G', 'G1')
dot.edge('G1', 'G2')
dot.edge('G2', 'G3', label='Yes')
dot.edge('G2', 'F', label='No')
dot.edge('G3', 'G4')
dot.edge('G4', 'A', label='Continue')

# Prediction and Evaluation
dot.node('H', 'Load Optimal Model')
dot.node('I', 'Perform Inference on Test/Prediction Data')
dot.node('J', 'Compare with Actual Stock Prices')
dot.node('K', 'Calculate Evaluation Metrics (MSE)')

# Add edges for prediction and evaluation
dot.edge('G4', 'H')
dot.edge('H', 'I')
dot.edge('I', 'J')
dot.edge('J', 'K')

# Render the graph to a file
dot.render('training_evaluation_workflow', format='png')

# Optionally, view the graph
# dot.view()