import pandas as pd
import numpy as np
import os
import glob
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

ruta_archivo_acciones = 'individual_stocks_5yr'
VOLVER_MENU = "\n\nPresiona ENTER para volver al menú..."

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def limpiar_consola():
    os.system('cls' if os.name == 'nt' else 'clear')

def cargar_datos_csv():
    archivos = glob.glob(os.path.join(ruta_archivo_acciones, '*.csv'))
    if not archivos:
        print(f"❌ No se encontraron archivos CSV en '{ruta_archivo_acciones}'")
        return None
    
    df = pd.concat(
        (pd.read_csv(f, engine='c', low_memory=False) for f in archivos),
        ignore_index=True
    )
    return df

def limpiar_datos(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    columnas_criticas = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df.dropna(subset=columnas_criticas)
    
    if 'Name' in df.columns:
        df = df.drop_duplicates(subset=['date', 'Name'], keep='first')
        df = df.sort_values(['Name', 'date'])
    else:
        df = df.sort_values('date')

    filtro_fecha = (df['date'] >= '2013-01-01') & (df['date'] <= '2020-12-31')
    df = df[filtro_fecha]
    df = df.reset_index(drop=True)
    
    return df

def crear_features_simple(df, dias_prediccion):
        
    def _crear_features_por_stock(stock_df):
        stock_df = stock_df.sort_values('date').copy()
        
        # Features básicas
        stock_df['retorno_1d'] = stock_df['close'].pct_change() * 100
        stock_df['retorno_5d'] = stock_df['close'].pct_change(5) * 100
        stock_df['retorno_10d'] = stock_df['close'].pct_change(10) * 100
        stock_df['retorno_20d'] = stock_df['close'].pct_change(20) * 100
        
        # Medias móviles
        stock_df['mm7'] = stock_df['close'].rolling(7, min_periods=7).mean()
        stock_df['mm21'] = stock_df['close'].rolling(21, min_periods=21).mean()
        stock_df['mm50'] = stock_df['close'].rolling(50, min_periods=50).mean()
        
        # Volatilidad
        stock_df['vol_7'] = stock_df['retorno_1d'].rolling(7, min_periods=7).std()
        stock_df['vol_21'] = stock_df['retorno_1d'].rolling(21, min_periods=21).std()
        
        # RSI simple
        diferencia = stock_df['close'].diff()
        ganancia = diferencia.where(diferencia > 0, 0.0)
        perdida = -diferencia.where(diferencia < 0, 0.0)
        media_ganancia = ganancia.rolling(14, min_periods=14).mean()
        media_perdida = perdida.rolling(14, min_periods=14).mean()
        fuerza_relativa = media_ganancia / (media_perdida + 1e-10)
        stock_df['rsi'] = 100 - (100 / (1 + fuerza_relativa))
        
        # OBJETIVO
        stock_df['precio_futuro'] = stock_df['close'].shift(-dias_prediccion)
        stock_df['objetivo'] = (stock_df['precio_futuro'] > stock_df['close']).astype(int)
        
        return stock_df
    
    if 'Name' in df.columns:
        grupos_features = []
        for nombre, stock_df in df.groupby('Name', sort=False):
            stock_features = _crear_features_por_stock(stock_df)
            stock_features['Name'] = nombre
            grupos_features.append(stock_features)
        if grupos_features:
            df_features = pd.concat(grupos_features, ignore_index=True)
        else:
            df_features = pd.DataFrame()
    else:
        df_features = _crear_features_por_stock(df)
    
    df_features = df_features.dropna().reset_index(drop=True)
    return df_features

# ═══════════════════════════════════════════════════════════════════════════
# OPCIÓN 1: COMPARACIÓN DE HORIZONTES
# ═══════════════════════════════════════════════════════════════════════════

def evaluar_horizonte(df, dias_prediccion, verbose=False):
    """Evalúa un horizonte temporal específico"""
    df_features = crear_features_simple(df, dias_prediccion)
    if df_features.empty:
        return None

    fecha_corte = df_features['date'].quantile(0.8)
    train_features = df_features[df_features['date'] < fecha_corte].copy()
    test_features = df_features[df_features['date'] >= fecha_corte].copy()
    if train_features.empty or test_features.empty:
        return None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f" EVALUANDO HORIZONTE: {dias_prediccion} DÍAS ".center(70))
        print(f"{'='*70}")
        print(f"Train: {train_features['date'].min().date()} → {train_features['date'].max().date()}")
        print(f"Test:  {test_features['date'].min().date()} → {test_features['date'].max().date()}")
    
    cols_drop = []
    for columna in ['objetivo', 'date', 'Name', 'precio_futuro']:
        if columna in train_features.columns:
            cols_drop.append(columna)
    X_train = train_features.drop(columns=cols_drop)
    y_train = train_features['objetivo']
    X_test = test_features.drop(columns=cols_drop)
    y_test = test_features['objetivo']
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    modelo = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    
    test_with_pred = test_features.copy()
    test_with_pred['prediccion'] = y_pred
    
    def calcular_retorno(stock_df, n_dias):
        stock_df = stock_df.sort_values('date')
        stock_df[f'retorno_{n_dias}d'] = (
            stock_df['close'].shift(-n_dias) / stock_df['close'] - 1
        )
        return stock_df
    
    if 'Name' in test_with_pred.columns:
        grupos_retornos = []
        for nombre, stock_df in test_with_pred.groupby('Name', sort=False):
            stock_retorno = calcular_retorno(stock_df, dias_prediccion)
            stock_retorno['Name'] = nombre
            grupos_retornos.append(stock_retorno)
        test_with_pred = pd.concat(grupos_retornos, ignore_index=True) if grupos_retornos else pd.DataFrame()
    else:
        test_with_pred = calcular_retorno(test_with_pred, dias_prediccion)
    
    mask = ~test_with_pred[f'retorno_{dias_prediccion}d'].isna()
    retornos = test_with_pred.loc[mask, f'retorno_{dias_prediccion}d']
    preds = test_with_pred.loc[mask, 'prediccion']
    
    estrategia_retornos = retornos[preds == 1]
    
    if len(estrategia_retornos) > 0:
        victorias = estrategia_retornos[estrategia_retornos > 0]
        perdidas = estrategia_retornos[estrategia_retornos < 0]
        
        ratio_victoria = len(victorias) / len(estrategia_retornos) if len(estrategia_retornos) > 0 else 0
        
        media_victoria = victorias.mean() if len(victorias) > 0 else 0
        media_perdida = perdidas.mean() if len(perdidas) > 0 else 0

        if len(perdidas) == 0:
            factor_beneficio = np.inf if len(victorias) > 0 else 0
        else:
            factor_beneficio = victorias.sum() / abs(perdidas.sum()) if len(victorias) > 0 else 0
        
        periodos_por_año = 252 / dias_prediccion
        rentabilidad = estrategia_retornos.mean() / estrategia_retornos.std() * np.sqrt(periodos_por_año) if estrategia_retornos.std() > 0 else 0
    else:
        ratio_victoria = 0
        factor_beneficio = 0
        rentabilidad = 0
        media_victoria = 0
        media_perdida = 0
    
    resultados = {
        'dias': dias_prediccion,
        'Precision': accuracy,
        'Precision_Balanceada': balanced_acc,
        'Ratio_Victoria': ratio_victoria,
        'Rentabilidad': rentabilidad,
        'Beneficio': factor_beneficio,
        'Media_Victoria': media_victoria,
        'Media_Perdida': media_perdida,
        'n_train': len(train_features),
        'n_test': len(test_features),
        'distribucion_clase_train': y_train.mean(),
        'distribucion_clase_test': y_test.mean(),
        'confusion_matrix': matriz_confusion.tolist()
    }
    
    if verbose:
        print("\n📊 RESULTADOS:")
        print(f"   Precision:            {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision_Balanceada:   {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
        print(f"   Ratio_Victoria:            {ratio_victoria:.4f} ({ratio_victoria*100:.2f}%)")
        print(f"   Rentabilidad:        {rentabilidad:.3f}")
        print(f"   Beneficio:       {factor_beneficio:.2f}")
        print(f"   Media_Victoria:             {media_victoria*100:+.3f}%")
        print(f"   Media_Perdida:            {media_perdida*100:+.3f}%")
        print(f"   Matriz de Confusión:    {matriz_confusion.tolist()}")
    
    return resultados

def opcion_comparar_horizontes(df):
    """Opción 1: Comparar diferentes horizontes temporales"""
    limpiar_consola()
    
    print("="*70)
    print(" COMPARACIÓN DE HORIZONTES TEMPORALES ".center(70))
    print("="*70)
    
    horizontes = [1, 3, 5, 7, 10, 15, 20]
    
    resultados_lista = []
    
    for dias in horizontes:
        print(f"\n⏳ Procesando horizonte {dias} días...")
        res = evaluar_horizonte(df, dias, verbose=False)
        if res is None:
            print("   ⚠️  Sin datos suficientes para evaluar este horizonte")
            continue
        resultados_lista.append(res)
        print(f"   ✓ Precision: {res['Precision']:.2%}, Precision_Balanceada: {res['Precision_Balanceada']:.2%}, Rentabilidad: {res['Rentabilidad']:.2f}, Beneficio: {res['Beneficio']:.2f}")

    if not resultados_lista:
        print("\n⚠️  No se pudieron calcular resultados para ningún horizonte")
        input(VOLVER_MENU)
        return
    
    df_resultados = pd.DataFrame(resultados_lista)
    
    print("\n" + "="*70)
    print(" TABLA COMPARATIVA ".center(70))
    print("="*70)
    print("\n" + df_resultados[['dias', 'Precision', 'Precision_Balanceada', 'Rentabilidad', 'Beneficio', 'Ratio_Victoria']].to_string(index=False))
    
    mejor_rentabilidad = df_resultados.loc[df_resultados['Rentabilidad'].idxmax()]
    
    print("\n" + "="*70)
    print(" RECOMENDACIÓN ".center(70))
    print("="*70)
    
    print(f"\n🏆 Mejor horizonte: {mejor_rentabilidad['dias']:.0f} días")
    print(f"   Precision:              {mejor_rentabilidad['Precision']:.2%}")
    print(f"   Precision_Balanceada:   {mejor_rentabilidad['Precision_Balanceada']:.2%}")
    print(f"   Rentabilidad:           {mejor_rentabilidad['Rentabilidad']:.3f}")
    print(f"   Beneficio:              {mejor_rentabilidad['Beneficio']:.2f}")
    
    if mejor_rentabilidad['Rentabilidad'] > 0.8 and mejor_rentabilidad['Precision_Balanceada'] > 0.52:
        print("\n✅ Este horizonte es viable para trading")
    else:
        print("\n⚠️  Resultados moderados, considerar mejoras")
    
    input(VOLVER_MENU)

# ═══════════════════════════════════════════════════════════════════════════
# OPCIÓN 2: ENTRENAR MODELO ESPECÍFICO
# ═══════════════════════════════════════════════════════════════════════════

def opcion_entrenar_modelo(df):
    """Opción 2: Entrenar modelo con horizonte específico"""
    limpiar_consola()
    
    print("="*70)
    print(" ENTRENAR MODELO ".center(70))
    print("="*70)
    
    print("\nHorizontes disponibles:")
    print("  1 = 1 día")
    print("  3 = 3 días")
    print("  5 = 5 días")
    print("  7 = 7 días")
    print("  10 = 10 días")
    print("  15 = 15 días")
    print("  20 = 20 días")
    
    while True:
        try:
            dias = int(input("\n¿Qué horizonte quieres usar?: "))
            if dias in [1, 3, 5, 7, 10, 15, 20]:
                break
            else:
                print("❌ Por favor elige un horizonte válido")
        except ValueError:
            print("❌ Entrada inválida")
    
    print(f"\n🔧 Entrenando modelo con horizonte de {dias} días...")
    
    print("\n   Creando features...", end=" ", flush=True)
    df_features = crear_features_simple(df, dias)
    print("✓")

    fecha_corte = df_features['date'].quantile(0.8)
    train_features = df_features[df_features['date'] < fecha_corte].copy()
    test_features = df_features[df_features['date'] >= fecha_corte].copy()

    print(f"   Train: {train_features['date'].min().date()} → {train_features['date'].max().date()}")
    print(f"   Test:  {test_features['date'].min().date()} → {test_features['date'].max().date()}")
    
    cols_drop = []
    for columna in ['objetivo', 'date', 'Name', 'precio_futuro']:
        if columna in train_features.columns:
            cols_drop.append(columna)
    X_train = train_features.drop(columns=cols_drop)
    y_train = train_features['objetivo']
    X_test = test_features.drop(columns=cols_drop)
    y_test = test_features['objetivo']
    
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print("   Entrenando XGBoost...", end=" ", flush=True)
    modelo = XGBClassifier(
        n_estimators=300,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    
    modelo.fit(X_train, y_train)
    print("✓")
    
    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    precision_Balanceada = balanced_accuracy_score(y_test, y_pred)
    matriz_confusion = confusion_matrix(y_test, y_pred)
    
    print(f"\n📊 Precision en test: {precision:.2%}")
    print(f"📊 Balanced Accuracy en test: {precision_Balanceada:.2%}")
    
    # Guardar modelo en carpeta dedicada por horizonte
    carpeta_modelo_base = 'Modelos'
    carpeta_modelo_dias = os.path.join(carpeta_modelo_base, f'modelo_{dias}_dias')
    os.makedirs(carpeta_modelo_dias, exist_ok=True)

    ruta_modelo = os.path.join(carpeta_modelo_dias, f'modelo_{dias}_dias.pkl')
    ruta_features = os.path.join(carpeta_modelo_dias, f'features_{dias}_dias.pkl')
    ruta_metricas = os.path.join(carpeta_modelo_dias, f'metricas_{dias}_dias.pkl')

    joblib.dump(modelo, ruta_modelo)
    joblib.dump(X_train.columns.tolist(), ruta_features)
    joblib.dump(
        {
            'precision_test': float(precision),
            'balanced_accuracy_test': float(precision_Balanceada),
            'confusion_matrix': matriz_confusion.tolist()
        },
        ruta_metricas
    )
    
    print("\n💾 Archivos guardados:")
    print(f"   - {ruta_modelo}")
    print(f"   - {ruta_features}")
    print(f"   - {ruta_metricas}")
    
    input(VOLVER_MENU)

# ═══════════════════════════════════════════════════════════════════════════
# OPCIÓN 3: PREDECIR ACCIÓN ESPECÍFICA
# ═══════════════════════════════════════════════════════════════════════════

def opcion_predecir_accion(df):
    """Opción 3: Predecir tendencia de una acción específica"""
    limpiar_consola()
    
    print("="*70)
    print(" PREDICCIÓN INDIVIDUAL DE ACCIÓN ".center(70))
    print("="*70)

    if 'Name' not in df.columns:
        print("\n❌ El dataset no contiene la columna 'Name' (nombre de la acción)")
        input(VOLVER_MENU)
        return
    
    # Verificar que hay modelos disponibles
    modelos_disponibles = []
    for dias in [1, 3, 5, 7, 10, 15, 20]:
        carpeta_modelo_dias = os.path.join('Modelos', f'modelo_{dias}_dias')
        ruta_modelo = os.path.join(carpeta_modelo_dias, f'modelo_{dias}_dias.pkl')
        ruta_features = os.path.join(carpeta_modelo_dias, f'features_{dias}_dias.pkl')
        if os.path.exists(ruta_modelo) and os.path.exists(ruta_features):
            modelos_disponibles.append(dias)
    
    if not modelos_disponibles:
        print("\n❌ No hay modelos entrenados disponibles.")
        print("   Primero debes entrenar un modelo (Opción 2)")
        input(VOLVER_MENU)
        return
    
    texto_modelos_disponibles = ""
    for indice, dias_disponibles in enumerate(modelos_disponibles):
        if indice > 0:
            texto_modelos_disponibles += ", "
        texto_modelos_disponibles += f"{dias_disponibles} días"
    print(f"\n📁 Modelos disponibles: {texto_modelos_disponibles}")
    
    while True:
        try:
            dias_modelo = int(input("\n¿Qué modelo quieres usar? (días): "))
            if dias_modelo in modelos_disponibles:
                break
            else:
                print(f"❌ Modelo no disponible. Elige entre: {modelos_disponibles}")
        except ValueError:
            print("❌ Entrada inválida")
    
    # Cargar modelo
    carpeta_modelo_dias = os.path.join('Modelos', f'modelo_{dias_modelo}_dias')
    ruta_modelo = os.path.join(carpeta_modelo_dias, f'modelo_{dias_modelo}_dias.pkl')
    ruta_features = os.path.join(carpeta_modelo_dias, f'features_{dias_modelo}_dias.pkl')
    ruta_metricas = os.path.join(carpeta_modelo_dias, f'metricas_{dias_modelo}_dias.pkl')

    modelo = joblib.load(ruta_modelo)
    feature_names = joblib.load(ruta_features)
    precision_modelo = None
    precision_balanceada_modelo = None

    if os.path.exists(ruta_metricas):
        metricas = joblib.load(ruta_metricas)
        precision_modelo = metricas.get('precision_test')
        precision_balanceada_modelo = metricas.get('balanced_accuracy_test')
    
    # Mostrar acciones disponibles
    acciones_disponibles = sorted(df['Name'].unique())
    
    print(f"\n📊 Acciones disponibles ({len(acciones_disponibles)}):")
    print("   Ejemplos: AAPL, MSFT, GOOGL, AMZN, TSLA, FB, NVDA...")
    
    while True:
        nombre = input("\n¿Qué acción quieres analizar? (nombre): ").strip().upper()
        if nombre in acciones_disponibles:
            break
        else:
            # Buscar coincidencias parciales
            coincidencias = [a for a in acciones_disponibles if nombre in a]
            if coincidencias:
                print(f"   ¿Quisiste decir? {', '.join(coincidencias[:10])}")
            else:
                print(f"   ❌ Acción no encontrada. Intenta otro nombre.")
    
    # Obtener datos de la acción
    stock_data = df[df['Name'] == nombre].copy()
    stock_data = stock_data.sort_values('date')
    
    print(f"\n🔍 Analizando {nombre}...")
    print(f"   Datos disponibles: {stock_data['date'].min().date()} → {stock_data['date'].max().date()}")
    
    # Crear features para la acción
    stock_features = crear_features_simple(stock_data, dias_modelo)
    
    if len(stock_features) == 0:
        print("❌ No hay suficientes datos para crear features")
        input(VOLVER_MENU)
        return
    
    # Usar solo las últimas N filas para predicción
    ultimos_datos = stock_features.tail(30)
    
    # Preparar features
    cols_drop = []
    for columna in ['objetivo', 'date', 'Name', 'precio_futuro']:
        if columna in ultimos_datos.columns:
            cols_drop.append(columna)
    X = ultimos_datos.drop(columns=cols_drop)
    
    # Asegurar que tenemos las mismas features
    X = X.reindex(columns=feature_names, fill_value=0)
    
    # Predecir
    predicciones = modelo.predict(X)
    probabilidades = modelo.predict_proba(X)
    
    # Obtener última predicción
    ultima_pred = predicciones[-1]
    ultima_prob = probabilidades[-1]
    
    # Datos actuales
    ultimo_precio = ultimos_datos['close'].iloc[-1]
    ultima_fecha = ultimos_datos['date'].iloc[-1]
    
    # Mostrar resultados
    print("\n" + "="*70)
    print(" ANÁLISIS DE PREDICCIÓN ".center(70))
    print("="*70)
    
    print(f"\n📅 Fecha del análisis: {ultima_fecha.date()}")
    print(f"💵 Precio actual: ${ultimo_precio:.2f}")
    print(f"🎯 Horizonte de predicción: {dias_modelo} días")
    
    print(f"\n{'='*70}")
    
    if ultima_pred == 1:
        print("📈 PREDICCIÓN: ALCISTA (precio subirá)")
        confianza = ultima_prob[1] * 100
        
    else:
        print("📉 PREDICCIÓN: BAJISTA (precio bajará)")
        confianza = ultima_prob[0] * 100
        
    
    print(f"\n{'='*70}")
    print(f"   Confianza: {confianza:.1f}%")
    print(f"   Probabilidad SUBE: {ultima_prob[1]*100:.1f}%")
    print(f"   Probabilidad BAJA: {ultima_prob[0]*100:.1f}%")
    print(f"{'='*70}")
    
    # Interpretación de la fuerza
    print(f"\n💡 INTERPRETACIÓN:")
    
    if confianza > 70:
        print(f"   ✅ SEÑAL FUERTE - Alta confianza en la predicción")
        if ultima_pred == 1:
            print(f"   → Considerar COMPRA significativa")
        else:
            print(f"   → Considerar VENTA o evitar comprar")
    elif confianza > 60:
        print(f"   ⚠️  SEÑAL MODERADA - Confianza media")
        if ultima_pred == 1:
            print(f"   → Considerar COMPRA pequeña/media")
        else:
            print(f"   → Precaución, posible caída")
    else:
        print(f"   ⚪ SEÑAL DÉBIL - Baja confianza, resultado incierto")
        print(f"   → NO operar, señal poco clara")
    
    # Contexto técnico
    print(f"\n📊 CONTEXTO TÉCNICO:")
    fuerza_relativa = ultimos_datos['rsi'].iloc[-1]
    mm7 = ultimos_datos['mm7'].iloc[-1]
    mm21 = ultimos_datos['mm21'].iloc[-1]
    vol_7 = ultimos_datos['vol_7'].iloc[-1]
    
    print(f"   Fuerza Relativa: {fuerza_relativa:.1f}", end="")
    if fuerza_relativa > 70:
        print(" (sobrecomprado ⚠️)")
    elif fuerza_relativa < 30:
        print(" (sobrevendido ⚠️)")
    else:
        print(" (neutral)")
    
    print(f"   MM7: ${mm7:.2f}")
    print(f"   MM21: ${mm21:.2f}")
    
    if mm7 > mm21:
        print(f"   Tendencia: ALCISTA (MM7 > MM21)")
    else:
        print(f"   Tendencia: BAJISTA (MM7 < MM21)")
    
    print(f"   Volatilidad (7d): {vol_7:.2f}%")
    
    # Historial de predicciones últimos 30 días
    print(f"\n📈 HISTORIAL ÚLTIMOS 30 DÍAS:")
    predicciones_hist = predicciones[-30:]
    alcistas = (predicciones_hist == 1).sum()
    bajistas = (predicciones_hist == 0).sum()
    
    print(f"   Señales alcistas: {alcistas}/30 ({alcistas/30*100:.0f}%)")
    print(f"   Señales bajistas: {bajistas}/30 ({bajistas/30*100:.0f}%)")
    
    if alcistas > 20:
        print(f"   → Tendencia fuertemente ALCISTA en últimos 30 días")
    elif bajistas > 20:
        print(f"   → Tendencia fuertemente BAJISTA en últimos 30 días")
    else:
        print(f"   → Tendencia MIXTA/LATERAL en últimos 30 días")
    
    # Disclaimer
    print(f"\n{'='*70}")
    print("Características del modelo ".center(70))
    print(f"{'='*70}")
    if precision_modelo is not None:
        print(f"   Precision del modelo: {precision_modelo:.2%}")
        if precision_balanceada_modelo is not None:
            print(f"   Precision Balanceada: {precision_balanceada_modelo:.2%}")
    print(f"{'='*70}")
    
    input(VOLVER_MENU)

# ═══════════════════════════════════════════════════════════════════════════
# OPCIÓN 4: ANÁLISIS GENERAL (MÚLTIPLES ACCIONES)
# ═══════════════════════════════════════════════════════════════════════════

def opcion_analisis_general(df):
    """Opción 4: Analizar múltiples acciones a la vez"""
    limpiar_consola()
    
    print("="*70)
    print(" ANÁLISIS GENERAL - TOP RECOMENDACIONES ".center(70))
    print("="*70)

    if 'Name' not in df.columns:
        print("\n❌ El dataset no contiene la columna 'Name' (ticker)")
        input(VOLVER_MENU)
        return
    
    # Verificar modelos
    modelos_disponibles = []
    for dias in [1, 3, 5, 7, 10, 15, 20]:
        carpeta_modelo_dias = os.path.join('Modelos', f'modelo_{dias}_dias')
        ruta_modelo = os.path.join(carpeta_modelo_dias, f'modelo_{dias}_dias.pkl')
        ruta_features = os.path.join(carpeta_modelo_dias, f'features_{dias}_dias.pkl')
        if os.path.exists(ruta_modelo) and os.path.exists(ruta_features):
            modelos_disponibles.append(dias)
    
    if not modelos_disponibles:
        print("\n❌ No hay modelos entrenados. Usa la Opción 2 primero.")
        input(VOLVER_MENU)
        return
    
    texto_modelos_disponibles = ""
    for indice, dias_disponibles in enumerate(modelos_disponibles):
        if indice > 0:
            texto_modelos_disponibles += ", "
        texto_modelos_disponibles += f"{dias_disponibles} días"
    print(f"\n📁 Modelos disponibles: {texto_modelos_disponibles}")
    
    while True:
        try:
            dias_modelo = int(input("\n¿Qué modelo quieres usar? (días): "))
            if dias_modelo in modelos_disponibles:
                break
        except ValueError:
            pass
    
    # Cargar modelo
    carpeta_modelo_dias = os.path.join('Modelos', f'modelo_{dias_modelo}_dias')
    ruta_modelo = os.path.join(carpeta_modelo_dias, f'modelo_{dias_modelo}_dias.pkl')
    ruta_features = os.path.join(carpeta_modelo_dias, f'features_{dias_modelo}_dias.pkl')

    modelo = joblib.load(ruta_modelo)
    feature_names = joblib.load(ruta_features)
    
    print(f"\n🔍 Analizando todas las acciones...")
    
    resultados = []
    
    for nombre in df['Name'].unique():
        stock_data = df[df['Name'] == nombre].copy().sort_values('date')
        
        if len(stock_data) < 50:
            continue
        
        stock_features = crear_features_simple(stock_data, dias_modelo)
        
        if len(stock_features) == 0:
            continue
        
        # Última fila
        ultima = stock_features.iloc[-1]
        cols_drop = []
        for columna in ['objetivo', 'date', 'Name', 'precio_futuro']:
            if columna in stock_features.columns:
                cols_drop.append(columna)
        X = stock_features.drop(columns=cols_drop).iloc[[-1]]
        X = X.reindex(columns=feature_names, fill_value=0)
        
        pred = modelo.predict(X)[0]
        prob = modelo.predict_proba(X)[0]
        
        resultados.append({
            'Nombre': nombre,
            'Precio': ultima['close'],
            'Predicción': 'SUBE' if pred == 1 else 'BAJA',
            'Confianza': prob[pred] * 100,
            'Prob_Sube': prob[1] * 100,
            'Fuerza relativa': ultima['rsi'],
            'Tendencia': 'Alcista' if ultima['mm7'] > ultima['mm21'] else 'Bajista'
        })
    
    if not resultados:
        print("\n⚠️  No se generaron resultados. Revisa datos/modelo y vuelve a intentar.")
        input(VOLVER_MENU)
        return

    df_resultados = pd.DataFrame(resultados)
    
    # Top compras (predicción SUBE con alta confianza)
    top_compras = df_resultados[
        (df_resultados['Predicción'] == 'SUBE') & 
        (df_resultados['Confianza'] > 60)
    ].sort_values('Confianza', ascending=False).head(15)
    
    # Top ventas (predicción BAJA con alta confianza)
    top_ventas = df_resultados[
        (df_resultados['Predicción'] == 'BAJA') & 
        (df_resultados['Confianza'] > 60)
    ].sort_values('Confianza', ascending=False).head(15)
    
    print("\n" + "="*70)
    print(f" TOP 15 COMPRAS (horizonte {dias_modelo} días) ".center(70))
    print("="*70)
    
    if len(top_compras) > 0:
        print(f"\n{'Nombre':<8} {'Precio':>10} {'Confianza':>12} {'RSI':>8} {'Tendencia':<10}")
        print("-"*60)
        for _, row in top_compras.iterrows():
            print(f"{row['Nombre']:<8} ${row['Precio']:>9.2f} {row['Confianza']:>11.1f}% "
                  f"{row['Fuerza relativa']:>8.1f} {row['Tendencia']:<10}")
    else:
        print("\n   No hay señales de compra con confianza > 60%")
    
    print("\n" + "="*70)
    print(f" TOP 15 VENTAS (horizonte {dias_modelo} días) ".center(70))
    print("="*70)
    
    if len(top_ventas) > 0:
        print(f"\n{'Nombre':<8} {'Precio':>10} {'Confianza':>12} {'Fuerza relativa':>8} {'Tendencia':<10}")
        print("-"*60)
        for _, row in top_ventas.iterrows():
            print(f"{row['Nombre']:<8} ${row['Precio']:>9.2f} {row['Confianza']:>11.1f}% "
                  f"{row['Fuerza relativa']:>8.1f} {row['Tendencia']:<10}")
    else:
        print("\n   No hay señales de venta con confianza > 60%")
    
    # Guardar a CSV
    archivo_salida = f'recomendaciones_{dias_modelo}dias.csv'
    df_resultados.to_csv(archivo_salida, index=False)
    print(f"\n💾 Resultados completos guardados en: {archivo_salida}")
    
    input(VOLVER_MENU)

# ═══════════════════════════════════════════════════════════════════════════
# MENÚ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def mostrar_menu():
    """Muestra el menú principal"""
    limpiar_consola()
    
    print("="*70)
    print(" SISTEMA DE PREDICCIÓN DE ACCIONES - XGBoost ".center(70))
    print("="*70)
    
    print("\n📊 OPCIONES DISPONIBLES:")
    print("\n   1. 🔬 Comparar horizontes temporales (1-20 días)")
    print("   2. 🎓 Entrenar modelo con horizonte específico")
    print("   3. 🎯 Predecir acción específica (individual)")
    print("   4. 📈 Análisis general (Top recomendaciones)")
    print("   5. ❌ Salir")
    
    print("\n" + "="*70)

def main():
    """Función principal con menú interactivo"""

    limpiar_consola()
    
    # Cargar datos al inicio (solo una vez)
    print("🔧 Cargando datos del mercado...")
    df = cargar_datos_csv()
    
    if df is None:
        print("❌ Error al cargar datos. Verifica la carpeta 'individual_stocks_5yr'")
        return
    
    df = limpiar_datos(df)
    if 'Name' in df.columns:
        print(f"✓ Datos cargados: {len(df):,} registros, {df['Name'].nunique()} acciones")
    else:
        print(f"✓ Datos cargados: {len(df):,} registros")
    print("✓ Rango temporal: {} a {}".format(df['date'].min().date(), df['date'].max().date()))
    
    input("\nPresiona ENTER para continuar...")
    
    while True:
        mostrar_menu()
        
        opcion = input("\nSelecciona una opción (1-5): ").strip()
        
        if opcion == '1':
            opcion_comparar_horizontes(df)
        
        elif opcion == '2':
            opcion_entrenar_modelo(df)
        
        elif opcion == '3':
            opcion_predecir_accion(df)
        
        elif opcion == '4':
            opcion_analisis_general(df)
        
        elif opcion == '5':
            limpiar_consola()
            print("\n" + "="*70)
            print(" ¡Hasta luego! ".center(70))
            print("="*70)
            print("\n   Desarrollado para Memoria de Prácticas II")
            print("   Universidad Nebrija - 2026")
            print("\n" + "="*70 + "\n")
            break
        
        else:
            print("\n❌ Opción inválida. Intenta de nuevo.")
            input("\nPresiona ENTER para continuar...")

if __name__ == "__main__":
    main()