using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace BrainBuilder
{
    public class Loss
    {

        public enum Type { 
            MSE,
            MAE,
            CrossEntropy,
        }

        private Func<Vector<double>, Vector<double>, double> _lossFn;
        private Func<Vector<double>, Vector<double>, Vector<double>> _lossFnDerivative;

        private Type _type;
        public Type LossType { 
            get => _type; 
            set => _type = value; 
        }

        public Loss(Loss.Type type = Type.MSE) {
            Console.WriteLine(Mse(Vector<double>.Build.Dense([1, 0]), Vector<double>.Build.Dense([0, 1])));
            switch (type) {
                case Type.MSE:
                    _lossFn = Mse;
                    _lossFnDerivative = MseD;
                    break;
                case Type.MAE:
                    _lossFn = Mae;
                    _lossFnDerivative = MaeD;
                    break;
                case Type.CrossEntropy:
                    _lossFn = CrossEntropy;
                    _lossFnDerivative = CrossEntropyD;
                    break;
            }
        }

        public double GetLoss(Vector<double> predicted, Vector<double> real) {
            return _lossFn(predicted, real);
        }

        public Vector<double> GetLossDerivative(Vector<double> predicted, Vector<double> real) { 
            return _lossFnDerivative(predicted, real);
        }

        private double Mse(Vector<double> predicted, Vector<double> real) {
            return Statistics.Mean((predicted - real).PointwisePower(2));
        }

        private Vector<double> MseD(Vector<double> predicted, Vector<double> real) {
            return (predicted - real) * (2.0 / predicted.Count);
        }

        private double Mae(Vector<double> predicted, Vector<double> real)
        {
            return Statistics.Mean((predicted - real).PointwiseAbs());
        }

        private Vector<double> MaeD(Vector<double> predicted, Vector<double> real)
        {
            return (predicted - real).PointwiseSign() / (predicted.Count);
        }

        private double CrossEntropy(Vector<double> predicted, Vector<double> real)
        {
            double epsilon = 1e-8;
            Vector<double> adjustedPred = Vector<double>.Build.Dense(predicted.Count);
            for(int i = 0; i < predicted.Count; i++)
            {
                adjustedPred[i] = Math.Clamp(predicted[i], epsilon, 1.0 - epsilon);
            }

            double loss = 0.0;
            for(int i = 0; i < predicted.Count; i++)
            {
                double pred = adjustedPred[i];
                double lab = real[i];
                loss -= lab * Math.Log(pred) + (1 - lab) * Math.Log(1 - pred);
            }
            return loss / predicted.Count;
        }

        private Vector<double> CrossEntropyD(Vector<double> predicted, Vector<double> real)
        {
            var epsilon = 1e-8;
            Vector<double> adjustedPred = Vector<double>.Build.Dense(predicted.Count);
            for(int i = 0; i < predicted.Count; i++)
            {
                adjustedPred[i] = Math.Clamp(predicted[i], epsilon, 1.0-epsilon);
            }
            return (adjustedPred - real) / (adjustedPred.PointwiseMultiply(1.0 - adjustedPred));
        }
    }
}
