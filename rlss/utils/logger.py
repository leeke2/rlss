from aenum import Enum, skip
from typing import Callable, Optional
from torch.utils.tensorboard import SummaryWriter


class InfoType(Enum):
    @skip
    class Episode(Enum):
        EpReward = 'ep_reward'
        EpDone = 'ep_done'
        BestObj = 'best_obj'

    @skip
    class Step(Enum):
        Update = 'update'
        LossQ1 = 'loss_q1'
        LossQ2 = 'loss_q2'
        LossPolicy = 'loss_p'
        LossAlpha = 'loss_alpha'
        StepReward = 'step_reward'
        Alpha = 'alpha'
        Entropy = 'entropy'
        BatchRecency = 'batch_recency'
        BatchRepeated = 'batch_repeated'
        ExperienceUtilization = 'experience_utilization'


class CallbackTrigger(Enum):
    @skip
    class Episode(Enum):
        BeforeStart = 1
        Start = 2
        End = 3
        LastEnd = 4

    @skip
    class Step(Enum):
        BeforeStart = 5
        End = 6

    @skip
    class EpStep(Enum):
        BeforeStart = 7
        End = 8
        LastEnd = 9


class LogEntry:
    def __init__(self):
        self._data = {}

    def __getattr__(self, key):
        return self._data[key]


class EpisodeEntry(LogEntry):
    def __init__(self, i_episode):
        super(EpisodeEntry, self).__init__()

        self._data = {
            'i_episode': i_episode,
            'ep_reward': 0,
            'ep_done': False
        }


class StepEntry(LogEntry):
    def __init__(self, i_episode, i_step, ep_i_step):
        super(StepEntry, self).__init__()

        self._data = {
            'i_episode': i_episode,
            'i_step': i_step,
            'ep_i_step': ep_i_step,
            'update': False
        }


class Logger:
    def __init__(self, name):
        # Global information
        self.i_episode = 0
        self.i_step = 0
        self.i_update = 0

        # Episode information
        self.ep_reward = 0
        self.ep_i_step = 0
        self.ep_done = False

        self._callbacks = {}
        self._episode_log = []
        self._step_log = []

        log_fn = f'runs/{name}'
        self.writer = SummaryWriter(log_fn)

    def _callback_parameters(self, trigger):
        t = type(trigger)

        if t is CallbackTrigger.Episode:
            return self.i_episode, self._episode_entry

        if t is CallbackTrigger.EpStep:
            return self.ep_i_step, self._step_entry

        if t is CallbackTrigger.Step:
            return self.i_step, self._step_entry

    def _reset_ep_logs(self) -> None:
        self.ep_reward = 0
        self.ep_i_step = 0
        self.ep_done = False

    def ep_counter(self, n_episodes):
        while True:
            self._callback(CallbackTrigger.Episode.BeforeStart)

            self._reset_ep_logs()
            self._episode_entry = EpisodeEntry(self.i_episode)

            self._callback(CallbackTrigger.Episode.Start)
            yield self.i_episode

            self._callback(CallbackTrigger.Episode.End)

            self.push(self._episode_entry)

            if self.i_episode == n_episodes - 1:
                self._callback(CallbackTrigger.Episode.LastEnd)
                return

            self.i_episode += 1

    def episode_done(self):
        self.ep_done = True

    def step_counter(self, max_steps_per_episode):
        while True:
            self._step_entry = StepEntry(self.i_episode, self.i_step, self.ep_i_step)

            self._callback(CallbackTrigger.Step.BeforeStart)
            self._callback(CallbackTrigger.EpStep.BeforeStart)
            yield self.i_step, self.ep_i_step

            self._callback(CallbackTrigger.Step.End)
            self._callback(CallbackTrigger.EpStep.End)

            self.push(self._step_entry)

            self.ep_i_step += 1
            self.i_step += 1

            if self.ep_i_step == max_steps_per_episode or self.ep_done:
                self._callback(CallbackTrigger.EpStep.LastEnd)
                return

    def _callback(self, trigger: CallbackTrigger) -> None:
        if trigger in self._callbacks:
            var, parameters = self._callback_parameters(trigger)

            for fingerprint, (fn, exactly, interval, offset) in self._callbacks[trigger].items():
                if var < offset:
                    continue

                if exactly is not None:
                    if var - offset == exactly:
                        fn(parameters)
                else:
                    if (var - offset) % interval == interval - 1:
                        fn(parameters)

    def register_callback(self,
                          trigger: CallbackTrigger,
                          callback: Callable,
                          exactly: Optional[int] = None,
                          interval: int = 1,
                          offset: int = 0) -> str:

        if trigger not in self._callbacks:
            self._callbacks[trigger] = {}

        fingerprint = hex(id(callback))[-5:]
        self._callbacks[trigger][fingerprint] = (callback, exactly, interval, offset)

        return fingerprint

    def deregister_callback(self, trigger: CallbackTrigger, fingerprint: str) -> None:
        if trigger in self._callbacks and fingerprint in self._callbacks[trigger]:
            del self._callbacks[trigger][fingerprint]

    def log(self, field, value=None):
        if field == InfoType.Step.Update:
            value = True
            self.i_update += 1

        if field == InfoType.Step.StepReward:
            self.ep_reward += value
            self._episode_entry._data[InfoType.Episode.EpReward.value] += value

        if type(field) is InfoType.Episode:
            self._episode_entry._data[field.value] = value
        else:
            self._step_entry._data[field.value] = value

    def push(self, log) -> None:
        t = type(log)

        if t is EpisodeEntry:
            self._episode_log.append(log)
            i_episode = log.i_episode

            for key, val in log._data.items():
                if key != 'i_episode':
                    self.writer.add_scalar(f'episode/{key}', val, i_episode)
        else:
            self._step_log.append(log)
            i_step = log.i_step

            for key, val in log._data.items():
                if key != 'i_step':
                    self.writer.add_scalar(f'step/{key}', val, i_step)

        self.writer.flush()
