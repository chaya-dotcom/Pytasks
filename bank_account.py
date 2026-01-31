class BankAccount:

    # constructor
    def __init__(self, name, balance):
        self.name = name
        self.balance = balance

    # deposit
    def deposit(self, amount):
        self.balance += amount
        print("Deposited:", amount)

    # withdraw
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print("Withdrawn:", amount)
        else:
            print("Low balance")

    # show balance
    def show_balance(self):
        print("Balance:", self.balance)


class SavingsAccount(BankAccount):

    # override
    def show_balance(self):
        print("Savings Balance:", self.balance)


# objects
acc1 = BankAccount("Chaya", 5000)
acc1.deposit(1000)
acc1.withdraw(2000)
acc1.show_balance()

acc2 = SavingsAccount("Anu", 8000)
acc2.deposit(500)
acc2.withdraw(1000)
acc2.show_balance()
