
"""
you need to implement an off-policy Monte-Carlo evaluation method. You need to write two different versions: (1) ordinary important sampling and (2) weighted importance sampling.
"""
class Policy(object):
    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            pi(a|s)
        """
        raise NotImplementedError()

    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError()
