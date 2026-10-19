function resultados = evaluar(yTrue, yPred, scores, classNames)
% EVALUAR  Evaluacion completa de un clasificador multiclase.
%
%   evaluar(yTrue, yPred, scores, classNames)
%
%   Calcula e imprime accuracy, precision/recall/F1 por clase y
%   macro-promediadas, AUC y AP (One-vs-Rest), y dibuja la matriz de
%   confusion. Ver docs/metrics.md para las definiciones.
%
%   Entradas:
%     yTrue      - etiquetas verdaderas (numericas 1..C o categorical)
%     yPred      - etiquetas predichas  (numericas 1..C o categorical)
%     scores     - matriz N x C de puntuaciones/probabilidades, con las
%                  columnas en el orden de las clases 1..C. Es la segunda
%                  salida de predict() o la salida de minibatchpredict().
%     classNames - cell array con los C nombres de clase
%
%   Salida (opcional):
%     resultados - struct con los campos accuracy, precision, recall, f1,
%                  auc, ap (vectores por clase) y las versiones macro.
%
%   Ejemplo:
%     [preds, scores] = predict(modelo, Xtest);
%     evaluar(y_test, preds, scores, class_names);

    yTrue = aIndice(yTrue);
    yPred = aIndice(yPred);
    nClases = numel(classNames);

    if size(scores, 2) ~= nClases
        error('evaluar: scores tiene %d columnas y hay %d clases.', ...
            size(scores, 2), nClases);
    end

    accuracy = mean(yTrue == yPred);

    C = confusionmat(yTrue, yPred, 'Order', 1:nClases);
    precision = zeros(nClases, 1);
    recall    = zeros(nClases, 1);
    f1        = zeros(nClases, 1);
    auc       = zeros(nClases, 1);
    ap        = zeros(nClases, 1);

    for c = 1:nClases
        TP = C(c, c);
        FP = sum(C(:, c)) - TP;
        FN = sum(C(c, :)) - TP;
        precision(c) = TP / max(TP + FP, eps);
        recall(c)    = TP / max(TP + FN, eps);
        f1(c) = 2 * precision(c) * recall(c) / max(precision(c) + recall(c), eps);

        [~, ~, ~, auc(c)] = perfcurve(yTrue, scores(:, c), c);
        % AP = area bajo la curva precision-recall. NOTA: perfcurve usa la
        % regla trapezoidal, mientras que average_precision_score de
        % scikit-learn interpola por escalones, asi que los dos valores no
        % coinciden exactamente. Ver docs/metrics.md.
        [~, ~, ~, ap(c)] = perfcurve(yTrue, scores(:, c), c, ...
            'XCrit', 'tpr', 'YCrit', 'ppv');
    end

    fprintf('\nAccuracy global: %.4f\n', accuracy);
    fprintf('Macro precision/recall/F1: %.3f / %.3f / %.3f\n', ...
        mean(precision), mean(recall), mean(f1));
    fprintf('AUC macro (One-vs-Rest):   %.3f\n', mean(auc));
    fprintf('AP macro (One-vs-Rest):    %.3f\n\n', mean(ap));

    fprintf('%-12s %-10s %-10s %-10s %-8s\n', ...
        'Clase', 'Precision', 'Recall', 'F1', 'Soporte');
    for c = 1:nClases
        fprintf('%-12s %-10.3f %-10.3f %-10.3f %-8d\n', ...
            classNames{c}, precision(c), recall(c), f1(c), sum(C(c, :)));
    end

    figure;
    confusionchart(yTrue, yPred, 'RowSummary', 'row-normalized');
    title(sprintf('Matriz de confusion (accuracy = %.3f)', accuracy));

    if nargout > 0
        resultados = struct( ...
            'accuracy', accuracy, ...
            'precision', precision, 'recall', recall, 'f1', f1, ...
            'auc', auc, 'ap', ap, ...
            'macroPrecision', mean(precision), ...
            'macroRecall', mean(recall), ...
            'macroF1', mean(f1), ...
            'macroAUC', mean(auc), ...
            'macroAP', mean(ap), ...
            'confusion', C);
    end
end

function v = aIndice(y)
    % Normaliza etiquetas a indices numericos 1..C en columna. Vale igual
    % para categorical (double() de un categorical devuelve el indice de
    % categoria) y para etiquetas numericas 1..C.
    v = double(y(:));
end
