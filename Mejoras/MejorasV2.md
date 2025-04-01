---
# Varias áreas de mejora y estrategias específicas para cada tarea:

## Para Daños (Dano):

1. Problemas principales:
    - Bajo recall en Fractura (9%) y Rayón (13%)
    - Precision muy baja en Desprendimiento (17%) y Rayón (11%)

2. Mejoras recomendadas:

### Ajustar pesos de clase más agresivamente
dano_weights = torch.tensor([
    1.5,  # Abolladura (mejorar recall)
    1.3,  # Arañazo
    3.0,  # Corrosión (no aparece en datos)
    2.5,  # Deformación
    5.0,  # Desprendimiento (prioridad)
    8.0,  # Fractura (máxima prioridad)
    7.0,  # Rayón
    1.2   # Rotura (ya funciona bien)
], device=device)

### Aumentar datos para clases minoritarias
transform_aumento_dano = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## Para Piezas:

1. Problemas principales:
    - Muchas clases con 0% de precisión/recall
    - Accuracy general muy bajo (20%)

2. Mejoras recomendadas:
### Estrategia 1: Agrupar piezas similares
### (Ejemplo - crear superclases)
    pieza_groups = {
        'Luces delanteras': [13, 14, 20, 21],
        'Luces traseras': [24, 25, 22, 23],
        'Guardabarros': [15, 16, 17, 18],
        # ... etc
    }

### Estrategia 2: Balancear datos
    pieza_weights = torch.ones(63, device=device)
    for idx in [5, 9, 10, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25]:
        pieza_weights[idx] = 3.0  # Aumentar peso para clases frecuentes

### Modificar criterion
    criterion['pieza'] = nn.CrossEntropyLoss(weight=pieza_weights)

### Aumento de datos específico para piezas
    transform_aumento_pieza = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

## Para Sugerencia:
- Precisión: 74% (Reparar) vs 56% (Reemplazar)
- Recall: 68% (Reparar) vs 62% (Reemplazar)
- F1-score: 71% (Reparar) vs 59% (Reemplazar)

### Problemas
- Desbalanceo de clases (más ejemplos de "Reparar")
- La sugerencia no aprovecha completamente el contexto de daños/piezas
- Pérdida de información entre tareas

1. Balanceo de Datos con Muestreo Estratificado
    - Desbalanceo a favor de "Reparar" (68% recall vs 62% en Reemplazar)

    ### Calcular pesos para el sampler
    sugerencia_counts = train_df['sugerencia'].value_counts()
    weights = 1.0 / sugerencia_counts[train_df['sugerencia']].values
    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True
    )

    ### Modificar DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Usar sampler en lugar de shuffle
        num_workers=4,
        pin_memory=True
    )


2. Head de Sugerencia Mejorado con Contexto
    ### Balancear el dataset
    sugerencia_weights = torch.tensor([1.0, 1.3], device=device)  # Más peso a Reemplazar
    criterion['sugerencia'] = nn.CrossEntropyLoss(weight=sugerencia_weights)

    class SugerenciaHead(nn.Module):
        def __init__(self, input_size, num_classes, dano_features, pieza_features):
            super().__init__()
            self.context_processor = nn.Sequential(
                nn.Linear(dano_features + pieza_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.main = nn.Sequential(
                nn.Linear(input_size + 128, 256),  # Combina features + contexto
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x, dano_probs, pieza_probs):
            context = torch.cat((dano_probs, pieza_probs), dim=1)
            context = self.context_processor(context)
            x = torch.cat((x, context), dim=1)
            return self.main(x)

3. Función de Pérdida Mejorada
    ### Focal Loss para manejar desbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            return focal_loss.mean()

    ### Configuración
    criterion['sugerencia'] = FocalLoss(alpha=torch.tensor([0.4, 0.6]))  # Más peso a "Reemplazar"

4. Regularización Específica
    ### Añadir al modelo principal
    self.sugerencia_dropout = nn.Dropout2d(0.2)  # Dropout espacial para features visuales

    ### Modificar el forward
    visual_features = self.sugerencia_dropout(base_features)
    sugerencia_out = self.sugerencia_head(visual_features, dano_probs, pieza_probs)

5. Entrenamiento con Métricas Balanceadas
    ### Métricas adicionales durante el entrenamiento
    def balanced_accuracy(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return np.mean(cm.diagonal() / cm.sum(axis=1))

    ### En el bucle de validación:
    sugerencia_preds = torch.cat(all_outputs['sugerencia'])
    sugerencia_true = torch.cat(all_labels['sugerencia'])
    bal_acc = balanced_accuracy(sugerencia_true.cpu(), sugerencia_preds.cpu())
    print(f"Balanced Accuracy: {bal_acc:.4f}")
---