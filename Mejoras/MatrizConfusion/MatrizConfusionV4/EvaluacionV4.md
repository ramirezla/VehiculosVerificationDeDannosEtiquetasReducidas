1. Evaluación de Daños (DANO):

    - Accuracy: 31.5% (Bajo)
    - Mejor clase: "Rotura" (F1=0.434)
    - Peor clase: "Fractura" (F1=0.067)

    - Problemas detectados:
        - Alto recall en "Abolladura" (40.2%) pero baja precisión (34%)
        - Clases minoritarias ("Rayón", "Fractura") con muy bajo desempeño
        - El modelo confunde frecuentemente entre "Abolladura" y "Rotura"

### Recomendaciones:

    # Aumentar capacidad del head de daños
    self.dano_head = nn.Sequential(
        nn.Linear(2048, 1536),  # Capa más grande
        nn.LayerNorm(1536),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2),  # Menor dropout
        nn.Linear(1536, 1024),
        nn.LeakyReLU(0.1),
        nn.Linear(1024, num_danos)
    )

2. Evaluación de Piezas (PIEZA):

    - Accuracy: 5.04% (Muy bajo)

    - Solo 2 clases muestran algún resultado:
        - "Luz indicadora delantera derecha" (recall=100% pero solo 6 muestras)
        - "Guardabarros trasero izquierdo" (F1=0.044)

    - Problemas graves:
        - 95% de las clases con 0 en todas las métricas
        - Desbalance extremo (algunas clases con 0 muestras en validación)

### Recomendaciones:

### Estrategias específicas para piezas:
1. Agrupar piezas similares (ej: espejos izquierdo/derecho -> "Espejos")
2. Implementar data augmentation específica:
   transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),  # Simular occlusiones
3. Usar focal loss más agresivo:
   'pieza': BalancedFocalLoss(alpha=pieza_weights, gamma=3)  # Gamma aumentado

3. Evaluación de Sugerencia:

    - Accuracy: 60.99% (Aceptable pero mejorable)

    - Sesgo evidente:
        - Alto recall para "Reparar" (90.1%)
        - Muy bajo recall para "Reemplazar" (15.8%)

    - Problema: El modelo tiende a predecir "Reparar" casi siempre

### Recomendaciones:

    # Balancear mejor el criterio de pérdida
    sugerencia_weights = torch.tensor([0.3, 0.7], device=device)  # Penalizar más errores en "Reemplazar"

    # Añadir métricas adicionales durante entrenamiento:
    def calculate_f1(outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        tp = ((preds == 1) & (labels == 1)).sum().float()
        fp = ((preds == 1) & (labels == 0)).sum().float()
        fn = ((preds == 0) & (labels == 1)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * (precision * recall) / (precision + recall + 1e-8)

### Diagnóstico General:

    - Problema Principal: Desbalance extremo en los datos, especialmente para piezas.

    - Evidencias:
        - Clases con 0 muestras en validación
        - Grandes diferencias en "support" entre clases
        - Métricas inconsistentes (ej: recall=100% con pocas muestras)

### Soluciones Propuestas:

    - Preprocesamiento de Datos:

    # Eliminar clases con menos de N muestras (ej: N=10)
    min_samples = 10
    valid_classes = [cls for cls, count in dataset.class_distribution['pieza'].items() if count >= min_samples]
    train_df = train_df[train_df['pieza'].isin(valid_classes)]

    - Arquitectura Mejorada:

    # Head de piezas con atención espacial
    class PiezaHead(nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            self.classifier = nn.Linear(in_features, num_classes)
            )
            
        def forward(self, x):
            attn_weights = F.softmax(self.attention(x), dim=1)
            features = (x * attn_weights).sum(dim=1)
            return self.classifier(features)

    - Estrategia de Entrenamiento:

    # Entrenamiento por etapas progresivas:
    1. Primero solo clases con >100 muestras
    2. Luego clases con >50 muestras
    3. Finalmente todas las clases con >10 muestras

    # Learning rate diferencial:
    optimizer = optim.AdamW([
        {'params': model.base_model.parameters(), 'lr': 1e-5},
        {'params': model.shared_features.parameters(), 'lr': 5e-4},
        {'params': model.pieza_head.parameters(), 'lr': 1e-3}
    ])

## Conclusión:

El modelo necesita:
    - Balanceo agresivo de datos (oversampling/undersampling)
    - Reducción de clases para piezas (agrupamiento)
    - Arquitectura especializada para piezas (mecanismos de atención)
    - Pérdidas focalizadas que penalicen más los errores en clases difíciles