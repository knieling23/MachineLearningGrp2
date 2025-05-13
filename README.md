# MachineLearningGrp2

Verbleibende Fehler sind analog zu Moodle Forum-Frage von Seyun Kim. Warten auf Rückmeldung von MP

Aktuelle Konsolenausgabe nach Ausführung der criterion.py-Datei:

(dvC_env) adrianadams@MacBook-Pro-AA MachineLearningGrp2 % python -m criterion
........F..F
======================================================================
FAIL: test__cross_entropy (__main__.TestClass_softmax_cross_entropy_loss.test__cross_entropy)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/adrianadams/Desktop/MachineLearningGrp2/criterion.py", line 127, in test__cross_entropy
    self.assertEqual(self.c._cross_entropy(i,c), o)
AssertionError: np.float64(-1.4076059644443804) != -1.2424533248940002

======================================================================
FAIL: test_gradient_data_loss (__main__.TestClass_softmax_cross_entropy_loss.test_gradient_data_loss)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/Users/adrianadams/Desktop/MachineLearningGrp2/criterion.py", line 152, in test_gradient_data_loss
    self.assertEqual(self.c.gradient_data_loss(i,c).tolist(), o.tolist())
AssertionError: Lists differ: [[0.0012332621856788756, -0.4987667378143211, 0.497533475628642[67 chars]843]] != [[0.0012332621856788754, -0.4987667378143211, 0.497533475628642[66 chars]484]]

First differing element 0:
[0.0012332621856788756, -0.4987667378143211, 0.49753347562864225]
[0.0012332621856788754, -0.4987667378143211, 0.49753347562864225]

- [[0.0012332621856788756, -0.4987667378143211, 0.49753347562864225],
?                       ^

+ [[0.0012332621856788754, -0.4987667378143211, 0.49753347562864225],
?                       ^

-  [3.072068425782561e-06, -0.49999692793157424, 0.49999385586314843]]
?                                                                  -

+  [3.072068425782561e-06, -0.49999692793157424, 0.4999938558631484]]

----------------------------------------------------------------------
Ran 12 tests in 0.392s

FAILED (failures=2)