#!/usr/bin/env python3
"""
Script de Avaliação de Modelos LSTM.

Gera um relatório completo de performance para apresentação acadêmica.

Uso:
    python scripts/evaluate_model.py AAPL
    python scripts/evaluate_model.py ALL --output reports/
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.data.data_loader import StockDataLoader
from src.data.preprocessor import StockDataPreprocessor
from src.models.lstm_model import LSTMPredictor
from src.training.improved_trainer import ImprovedLSTMPredictor
from config.settings import MODELS_DIR


def load_model_and_preprocessor(symbol: str):
    """Carrega modelo e preprocessor salvos."""
    model_path = MODELS_DIR / f"lstm_model_{symbol}.pth"
    scaler_path = MODELS_DIR / f"scaler_{symbol}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")
    
    # Tentar carregar como modelo melhorado primeiro
    try:
        model = ImprovedLSTMPredictor.load(model_path)
        logger.info(f" Carregado como ImprovedLSTMPredictor")
    except Exception:
        # Fallback para modelo original
        model = LSTMPredictor.load(model_path)
        logger.info(f" Carregado como LSTMPredictor (original)")
    
    preprocessor = StockDataPreprocessor.load(scaler_path)
    
    return model, preprocessor


def evaluate_model(symbol: str, test_days: int = 60) -> dict:
    """
    Avalia modelo com dados recentes.
    
    Args:
        symbol: Ticker da ação
        test_days: Dias de teste (dados mais recentes)
        
    Returns:
        Dict com métricas detalhadas
    """
    logger.info(f" Avaliando modelo para {symbol}")
    
    # Carregar modelo
    model, preprocessor = load_model_and_preprocessor(symbol)
    
    # Carregar dados recentes
    loader = StockDataLoader()
    df = loader.load_stock_data(symbol, "2024-01-01", datetime.now().strftime("%Y-%m-%d"))
    loader.validate_data(df)
    
    # Preparar features
    df_feat = preprocessor.prepare_features(df)
    
    # Escalar features
    feature_cols = preprocessor.feature_columns
    X_scaled = preprocessor.scaler.transform(df_feat[feature_cols].values)
    
    # Criar sequências para os últimos N dias
    sequence_length = preprocessor.sequence_length
    predictions = []
    actuals = []
    dates = []
    
    for i in range(sequence_length, len(X_scaled) - 1):
        # Sequência de entrada
        X_seq = X_scaled[i - sequence_length:i].reshape(1, sequence_length, -1)
        
        # Previsão
        pred_scaled = model.predict(X_seq)[0]
        pred_price = preprocessor.inverse_transform_target(pred_scaled)
        
        # Valor real (próximo dia)
        actual_price = df_feat['close'].iloc[i + 1]
        
        predictions.append(pred_price)
        actuals.append(actual_price)
        dates.append(df_feat['timestamp'].iloc[i + 1])
    
    # Converter para arrays
    predictions = np.array(predictions[-test_days:])
    actuals = np.array(actuals[-test_days:])
    dates = dates[-test_days:]
    
    # Calcular métricas
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    percent_errors = abs_errors / actuals * 100
    
    # Métricas principais
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(abs_errors)
    mape = np.mean(percent_errors)
    
    # R²
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Acurácia direcional
    actual_direction = np.diff(actuals) > 0
    pred_direction = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Métricas de variação
    actual_returns = np.diff(actuals) / actuals[:-1] * 100
    pred_returns = np.diff(predictions) / predictions[:-1] * 100
    return_correlation = np.corrcoef(actual_returns, pred_returns)[0, 1]
    
    # Estatísticas dos erros
    error_stats = {
        'min': round(float(np.min(abs_errors)), 2),
        'max': round(float(np.max(abs_errors)), 2),
        'std': round(float(np.std(abs_errors)), 2),
        'median': round(float(np.median(abs_errors)), 2),
        'p25': round(float(np.percentile(abs_errors, 25)), 2),
        'p75': round(float(np.percentile(abs_errors, 75)), 2),
        'p90': round(float(np.percentile(abs_errors, 90)), 2),
    }
    
    # Previsões vs reais (amostra)
    sample_predictions = [
        {
            'date': str(dates[i])[:10] if hasattr(dates[i], 'strftime') else str(dates[i])[:10],
            'predicted': round(predictions[i], 2),
            'actual': round(actuals[i], 2),
            'error': round(abs_errors[i], 2),
            'error_percent': round(percent_errors[i], 2)
        }
        for i in range(-10, 0)  # Últimas 10 previsões
    ]
    
    return {
        'symbol': symbol,
        'evaluation_date': datetime.now().isoformat(),
        'test_samples': len(predictions),
        'metrics': {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'mape': round(mape, 2),
            'r2': round(r2, 4),
            'directional_accuracy': round(directional_accuracy, 2),
            'return_correlation': round(return_correlation, 4) if not np.isnan(return_correlation) else 0
        },
        'error_distribution': error_stats,
        'price_range': {
            'min': round(float(np.min(actuals)), 2),
            'max': round(float(np.max(actuals)), 2),
            'mean': round(float(np.mean(actuals)), 2)
        },
        'sample_predictions': sample_predictions
    }


def generate_report(results: dict, output_path: Path = None):
    """Gera relatório de avaliação."""
    
    report = []
    report.append("=" * 70)
    report.append(" RELATÓRIO DE AVALIAÇÃO DO MODELO LSTM")
    report.append("=" * 70)
    report.append(f"Data: {results['evaluation_date'][:10]}")
    report.append(f"Símbolo: {results['symbol']}")
    report.append(f"Amostras de teste: {results['test_samples']}")
    report.append("")
    
    report.append(" MÉTRICAS DE PERFORMANCE:")
    report.append("-" * 40)
    m = results['metrics']
    report.append(f"  RMSE (Root Mean Square Error): ${m['rmse']}")
    report.append(f"  MAE (Mean Absolute Error): ${m['mae']}")
    report.append(f"  MAPE (Mean Absolute % Error): {m['mape']}%")
    report.append(f"  R² (Coef. Determinação): {m['r2']}")
    report.append(f"  Acurácia Direcional: {m['directional_accuracy']}%")
    report.append(f"  Correlação de Retornos: {m['return_correlation']}")
    report.append("")
    
    report.append(" INTERPRETAÇÃO:")
    report.append("-" * 40)
    
    # Interpretar MAPE
    if m['mape'] < 5:
        mape_interpretation = "Excelente (<5%)"
    elif m['mape'] < 10:
        mape_interpretation = "Bom (5-10%)"
    elif m['mape'] < 20:
        mape_interpretation = "Aceitável (10-20%)"
    else:
        mape_interpretation = "Precisa melhoria (>20%)"
    report.append(f"  MAPE: {mape_interpretation}")
    
    # Interpretar acurácia direcional
    if m['directional_accuracy'] > 60:
        dir_interpretation = "Muito bom (>60%)"
    elif m['directional_accuracy'] > 55:
        dir_interpretation = "Bom (55-60%)"
    elif m['directional_accuracy'] > 50:
        dir_interpretation = "Levemente melhor que random (50-55%)"
    else:
        dir_interpretation = "Não melhor que random (<50%)"
    report.append(f"  Acurácia Direcional: {dir_interpretation}")
    report.append("")
    
    report.append(" DISTRIBUIÇÃO DOS ERROS:")
    report.append("-" * 40)
    e = results['error_distribution']
    report.append(f"  Mínimo: ${e['min']:.2f}")
    report.append(f"  Percentil 25: ${e['p25']:.2f}")
    report.append(f"  Mediana: ${e['median']:.2f}")
    report.append(f"  Percentil 75: ${e['p75']:.2f}")
    report.append(f"  Percentil 90: ${e['p90']:.2f}")
    report.append(f"  Máximo: ${e['max']:.2f}")
    report.append(f"  Desvio Padrão: ${e['std']:.2f}")
    report.append("")
    
    report.append(" PREVISÕES vs VALORES REAIS (últimas 10):")
    report.append("-" * 40)
    report.append(f"{'Data':<12} {'Previsto':>10} {'Real':>10} {'Erro':>8} {'%':>6}")
    report.append("-" * 50)
    for p in results['sample_predictions']:
        report.append(
            f"{p['date']:<12} ${p['predicted']:>8.2f} ${p['actual']:>8.2f} "
            f"${p['error']:>6.2f} {p['error_percent']:>5.1f}%"
        )
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    # Imprimir
    print(report_text)
    
    # Salvar se output_path fornecido
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar texto
        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write(report_text)
        
        # Salvar JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f" Relatório salvo em {output_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Avaliar modelos LSTM")
    parser.add_argument("symbols", nargs="+", help="Símbolos para avaliar")
    parser.add_argument("--output", "-o", help="Diretório de saída para relatórios")
    parser.add_argument("--test-days", type=int, default=60, help="Dias para teste")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output) if args.output else None
    
    for symbol in args.symbols:
        symbol = symbol.upper()
        try:
            results = evaluate_model(symbol, args.test_days)
            
            output_path = None
            if output_dir:
                output_path = output_dir / f"evaluation_{symbol}"
            
            generate_report(results, output_path)
            
        except FileNotFoundError as e:
            logger.error(f" {e}")
        except Exception as e:
            logger.error(f" Erro ao avaliar {symbol}: {e}")


if __name__ == "__main__":
    main()

