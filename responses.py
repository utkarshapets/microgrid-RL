import numpy as np
import pandas as pd


class Person1:
    """
    This person uses energy consistently throughout the day, usually on computer, 
    apart from lunch between 12-1PM. They're the type of person that is mostly driven
    by points, so if the energy_unit_cost for that hour is high compared to the other 
    hours, their energy will be shifted relatively drastically from the baseline. """

    # TODO: analyze baseline, see if it's producing realistic values

    BASELINE_ENERGY = 300
    STARTING_POINTS = 500

    available_hours = [
        "08:00",
        "09:00",
        "10:00",
        "11:00",
        "12:00",
        "13:00",
        "14:00",
        "15:00",
        "16:00",
        "17:00",
        "18:00",
    ]

    available_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    def __init__(
        self, energy_cost_csv,
    ):
        self.energy_cost = pd.read_csv(energy_cost_csv)

    def baseline(self, day, hour):
        """Baseline from 12-1PM is reduced, as they're not using their computer and lights. 
        The rest of the makeup of the function is based on a qualitative assessment of relative
        levels of productivity throughout the day. The specific values can be changed. """

        # TODO: Verify values based on research into energy usage per day

        day_multiplier = {
            "Monday": 1.1,
            "Tuesday": 1.15,
            "Wednesday": 1,
            "Thursday": 0.9,
            "Friday": 0.8,
        }
        hour_multiplier = {
            "08:00": 0.8,
            "09:00": 0.9,
            "10:00": 1,
            "11:00": 0.9,
            "12:00": 0,
            "13:00": 0.9,
            "14:00": 1.1,
            "15:00": 1.1,
            "16:00": 1.0,
            "17:00": 0.9,
            "18:00": 0.8,
        }

        return self.BASELINE_ENERGY * day_multiplier[day] * hour_multiplier[hour]

    def total_energy_consumed(self, day):
        return sum([self.baseline(day, hour) for hour in self.available_hours])

    def redistributed_energy(self, day, points):
        self.MAX_DIFFERENTIAL = 5
        import cvxpy as cvx

        energy_curve = cvx.Variable(11)
        objective = cvx.Minimize(energy_curve.T * points["Cost"])
        constraints = [
            cvx.sum(energy_curve, axis=0, keepdims=True)
            == self.total_energy_consumed(day)
        ]
        for hour in range(11):
            constraints += [energy_curve[hour] >= 0]

        for hour in range(1, 11):
            constraints += [
                cvx.abs(energy_curve[hour] - energy_curve[hour - 1])
                <= self.MAX_DIFFERENTIAL
            ]

        problem = cvx.Problem(objective, constraints)
        problem.solve()
        return energy_curve.value

    def predicted_energy_behavior(self, day):

        self.AFFINITY_TO_POINTS = 0.8
        self.ENERGY_STD_DEV = 5

        perfect_energy_use = self.redistributed_energy(day, self.energy_cost)
        baseline_energy_use = [
            self.baseline(day, hour) for hour in self.available_hours
        ]
        means = np.empty(len(perfect_energy_use))
        for i in range(len(perfect_energy_use)):
            lesser, greater = (
                (perfect_energy_use[i], baseline_energy_use[i])
                if perfect_energy_use[i] < baseline_energy_use[i]
                else (baseline_energy_use[i], perfect_energy_use[i])
            )
            means[i] = lesser + 0.8 * (greater - lesser)
        return np.random.normal(means, self.ENERGY_STD_DEV)


class Person2(Person1):
    """
    This person comes into the office very sporadically on any given day, there's a
    random (Bernoulli) variable denoting if they came to the ofice that day or no. If 
    they do come, then they have fairly uniform usage each day. However, on days in which
    they did not come, that will result in a usage of the baseline energy regardless. On
    days they do come, they react similarly to the first person. 
    """

    BACKGROUND_ENERGY_USAGE = Person1.BASELINE_ENERGY / 10
    P_NOT_COME_TO_WORK = 0.2

    # TODO: test if above assumption is valid

    def predicted_energy_behavior(self, day):

        self.AFFINITY_TO_POINTS = 0.8
        self.ENERGY_STD_DEV = 5

        yes = np.random.binomial(1, self.P_NOT_COME_TO_WORK)

        if yes:
            return np.random.normal(
                [self.BACKGROUND_ENERGY_USAGE] * 11, self.ENERGY_STD_DEV
            )

        perfect_energy_use = self.redistributed_energy(day, self.energy_cost)
        baseline_energy_use = [
            self.baseline(day, hour) for hour in self.available_hours
        ]
        means = np.empty(len(perfect_energy_use))
        for i in range(len(perfect_energy_use)):
            lesser, greater = (
                (perfect_energy_use[i], baseline_energy_use[i])
                if perfect_energy_use[i] < baseline_energy_use[i]
                else (baseline_energy_use[i], perfect_energy_use[i])
            )
            means[i] = lesser + 0.8 * (greater - lesser)
        return np.random.normal(means, self.ENERGY_STD_DEV)


class Person3:
    """
    This person doesn't care much about the game in general, so their energy saving behavior 
    is aunffected, no matter what the points are (this is an edge case)
    """

    def __init__(self):
        pass

    def process_points(self, points_df):
        pass


class Person4:
    """
    This person is one who is less driven by points than the first person. but has higher 
    self-efficacy, so she is less affected by the change in points than the average person.
    """

    def __init__(self, behavior, self_efficacy):
        "Self efficacy is measured as a number between 0 and 1."
        self.behavior = behavior


if __name__ == "__main__":
    # person = Person4()
    # person = Person1()
    person = Person2("energy_cost.csv")
    print(person.predicted_energy_behavior("Monday"))
