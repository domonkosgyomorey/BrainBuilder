namespace BrainBuilder
{
    public static class LearningRateHandlerFactory
    {
        public static MomentumLearningRateHandler Create(
            LrHandlerType type,
            double baseRate,
            double momentum = 0.9,
            double decayRate = 0.7,
            int decayStep = 1000,
            int totalStep = 10000)
        {
            ILearningRateStrategy strategy = type switch
            {
                LrHandlerType.StepDecay => new StepDecayStrategy(),
                LrHandlerType.ExpDecay => new ExponentialDecayStrategy(),
                LrHandlerType.TimeDecay => new TimeBasedDecayStrategy(),
                LrHandlerType.CosineAnnealing => new CosineAnnealingStrategy(),
                LrHandlerType.None => new NoneStrategy(),
                _ => throw new ArgumentException("Unknown learning rate type")
            };

            return new MomentumLearningRateHandler(strategy, baseRate, momentum, decayRate, decayStep, totalStep);
        }
    }
}
