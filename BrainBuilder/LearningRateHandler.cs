namespace BrainBuilder
{
    public enum LrHandlerType
    {
        StepDecay,
        ExpDecay ,
        TimeDecay,
        CosineAnnealing,
        None
}

    public interface ILearningRateStrategy
    {
        double GetLearningRate(double baseRate, double iteration, double decayRate, int decayStep, int totalStep);
    }

    public class StepDecayStrategy:ILearningRateStrategy
    {
        public double GetLearningRate(double baseRate, double iteration, double decayRate, int decayStep, int totalStep) =>
            baseRate * Math.Pow(decayRate, iteration / decayStep);
    }

    public class ExponentialDecayStrategy:ILearningRateStrategy
    {
        public double GetLearningRate(double baseRate, double iteration, double decayRate, int decayStep, int totalStep) =>
            baseRate * Math.Exp(-decayRate * iteration);
    }

    public class TimeBasedDecayStrategy:ILearningRateStrategy
    {
        public double GetLearningRate(double baseRate, double iteration, double decayRate, int decayStep, int totalStep) =>
            baseRate / (1 + decayRate * iteration);
    }

    public class CosineAnnealingStrategy:ILearningRateStrategy
    {
        public double GetLearningRate(double baseRate, double iteration, double decayRate, int decayStep, int totalStep) =>
            baseRate * (1 + Math.Cos(Math.PI * iteration / totalStep)) / 2;
    }

    public class NoneStrategy:ILearningRateStrategy
    {
        public double GetLearningRate(double baseRate, double iteration, double decayRate, int decayStep, int totalStep)
        {
            return baseRate;
        }
    }

    public class MomentumLearningRateHandler
    {
        private readonly ILearningRateStrategy _strategy;
        private readonly double _momentum;
        private double _previousLearningRate;
        private double _baseRate;
        private double _decayRate;
        private int _decayStep;
        private int _totalStep;
        private double _iteration;

        public ILearningRateStrategy Strategy
        {
            get => _strategy;
        }

        public MomentumLearningRateHandler(ILearningRateStrategy strategy, double baseRate, double momentum, double decayRate, int decayStep, int totalStep)
        {
            _strategy = strategy;
            _momentum = momentum;
            _baseRate = baseRate;
            _decayRate = decayRate;
            _decayStep = decayStep;
            _totalStep = totalStep;
            _previousLearningRate = baseRate;
        }

        public double GetLearningRate()
        {
            double currentLearningRate = _strategy.GetLearningRate(_baseRate, _iteration, _decayRate, _decayStep, _totalStep);
            double adjustedLearningRate = _momentum * _previousLearningRate + (1 - _momentum) * currentLearningRate;
            _previousLearningRate = adjustedLearningRate;
            _iteration += 1;
            return adjustedLearningRate;
        }

        public double GetBaseRate() => _baseRate;
        public double GetMomentum() => _momentum;
        public double GetDecayRate() => _decayRate;
        public int GetDecayStep() => _decayStep;
        public int GetTotalStep() => _totalStep;
        public double GetIteration() => _iteration;
        public string GetStrategyType() => _strategy.GetType().Name;
    }
}
