# network.py サマリー

## 何を実装しているか

**多層パーセプトロン（MLP）を NumPy のみで実装した分類器**

- 入力 784次元（28×28 画像のフラット化）→ 隠れ層 → 10クラス出力
- 順伝播・逆伝播・Adam 最適化をすべて手書きで実装

---

## 関数 / クラス一覧
| 種類 | 名前 | 何をしているか |
|------|------|---------------|
| 関数 | `_softmax(x)` | 数値安定版 softmax（出力層の活性化） |
| 関数 | `_one_hot(labels, num_classes)` | ラベルを one-hot ベクトルに変換（損失計算用） |
| クラス | `NetworkConfig` | ハイパーパラメータをまとめる dataclass |
| クラス | `SimpleMLP` | MLP 本体（学習・推論・保存・読込を一括管理） |
| メソッド | `SimpleMLP._forward(x)` | 順伝播。線形出力とアクティベーション全層を返す |
| メソッド | `SimpleMLP.predict_proba(x)` | 各クラスの確率（softmax 出力）を返す |
| メソッド | `SimpleMLP.predict(x)` | 予測クラス番号（argmax）を返す |
| メソッド | `SimpleMLP.evaluate_accuracy(x, y)` | バッチ単位で正解率を計算 |
| メソッド | `SimpleMLP._adam_update(key, grad, lr, ...)` | Adam の m/v 更新と重みの更新 |
| メソッド | `SimpleMLP.train_epoch(x, y, epoch)` | 1エポック分の学習（シャッフル→ミニバッチ→逆伝播） |
| メソッド | `SimpleMLP.to_state()` | モデルを dict に変換して保存できる形式に |
| クラスメソッド | `SimpleMLP.from_state(state)` | dict からモデルを復元（後方互換あり） |

---

## 調整要素
| パラメータ | 場所 | デフォルト | 意味・影響 |
|-----------|------|-----------|-----------|
| `hidden_sizes` | `NetworkConfig` | `(256, 128)` | 隠れ層のユニット数と層数。増やすと表現力↑・過学習リスク↑ |
| `learning_rate` | `NetworkConfig` | `0.001` | Adam の学習率。大きいと発散、小さいと収束遅い |
| `batch_size` | `NetworkConfig` | `256` | ミニバッチサイズ。小さいと汎化↑・学習遅い、大きいと逆 |
| `seed` | `NetworkConfig` | `42` | 乱数シード。重み初期化とエポックごとのシャッフルに使用 |
| `input_size` | `NetworkConfig` | `784` | 入力次元数（Fashion-MNIST は固定） |
| `output_size` | `NetworkConfig` | `10` | クラス数（Fashion-MNIST は固定） |
| `beta1` | `_adam_update` の引数 | `0.9` | Adam の一次モーメント減衰率 |
| `beta2` | `_adam_update` の引数 | `0.999` | Adam の二次モーメント減衰率 |
| `eps` | `_adam_update` の引数 | `1e-8` | Adam のゼロ除算防止項 |

### 固定されている設計選択（変えるにはコード修正が必要）

- 隠れ層の活性化関数: **ReLU**（`np.maximum(0, z)`）
- 出力層の活性化関数: **Softmax**
- 損失関数: **Cross-Entropy Loss**
- 重み初期化: 隠れ層は **He 初期化**、出力層は **Xavier 初期化**
- 最適化アルゴリズム: **Adam**（SGD/RMSProp には非対応）
